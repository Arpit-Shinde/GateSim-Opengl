// GateSim parallel-frontier demo (wide + moderately deep circuit).
// Node 0 = source s. Layers 1..L, each with W nodes.
// Node (layer l, lane j) = XOR(prev, 0), where prev = s for l=1, else node (l-1, j).
// Toggling s gives a frontier of W nodes for L steps: W*L evaluations in total.

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <iostream>
#include <vector>
#include <cstdint>
#include <cstdlib>
#include <chrono>
#include <algorithm>

#include "frontier_shaders.h"

struct Node { int32_t type, in0, in1, pad; };   // matches ivec4 in the shader

using Clock = std::chrono::high_resolution_clock;
static double msSince(Clock::time_point t) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}

// ---------------- circuit construction ----------------
static void buildCircuit(int W, int L,
                         std::vector<Node>& nodes,
                         std::vector<uint32_t>& fanStart,
                         std::vector<uint32_t>& fanList)
{
    const uint32_t N = (uint32_t)W * (uint32_t)L;       // gate count; ids 1..N
    nodes.assign(N + 1, {0, -1, -1, 0});                // node 0 = source (type 0)

    for (uint32_t id = 1; id <= N; ++id) {
        int32_t prev = (id <= (uint32_t)W) ? 0 : (int32_t)(id - W);
        nodes[id] = {1, prev, -1, 0};                   // XOR(prev, unconnected=0)
    }

    fanStart.assign(N + 2, 0);
    fanList.reserve(N + W);

    for (int k = 1; k <= W; ++k) fanList.push_back(k);  // source -> layer 1
    fanStart[1] = (uint32_t)fanList.size();

    for (uint32_t id = 1; id <= N; ++id) {              // gate -> same lane, next layer
        if (id + W <= N) fanList.push_back(id + W);
        fanStart[id + 1] = (uint32_t)fanList.size();
    }
}

// ---------------- CPU reference (same two-phase algorithm, reused buffers) ----------------
struct CpuSim {
    const std::vector<Node>& nodes;
    const std::vector<uint32_t>& fanStart;
    const std::vector<uint32_t>& fanList;
    std::vector<uint32_t> val, marks, cur, nxt, res;
    uint32_t stamp = 0;

    CpuSim(const std::vector<Node>& n, const std::vector<uint32_t>& fs, const std::vector<uint32_t>& fl)
        : nodes(n), fanStart(fs), fanList(fl), val(n.size(), 0), marks(n.size(), 0) {}

    int propagate(std::vector<uint32_t>* sizes = nullptr) {
        val[0] ^= 1u;                                    // toggle source
        cur.assign(fanList.begin() + fanStart[0], fanList.begin() + fanStart[1]);
        int step = 0;
        while (!cur.empty()) {
            ++step;
            if (sizes) sizes->push_back((uint32_t)cur.size());

            res.resize(cur.size());                      // phase 1: read committed state only
            for (size_t i = 0; i < cur.size(); ++i) {
                const Node& nd = nodes[cur[i]];
                uint32_t a = nd.in0 >= 0 ? val[nd.in0] : 0u;
                uint32_t b = nd.in1 >= 0 ? val[nd.in1] : 0u;
                uint32_t old = val[cur[i]];
                uint32_t r = (nd.type == 1) ? (a ^ b) : old;
                res[i] = r | ((r != old) << 1);
            }

            nxt.clear();                                 // phase 2: commit together
            ++stamp;
            for (size_t i = 0; i < cur.size(); ++i) {
                if (!(res[i] & 2u)) continue;
                uint32_t id = cur[i];
                val[id] = res[i] & 1u;
                for (uint32_t k = fanStart[id]; k < fanStart[id + 1]; ++k) {
                    uint32_t o = fanList[k];
                    if (marks[o] != stamp) { marks[o] = stamp; nxt.push_back(o); }
                }
            }
            cur.swap(nxt);
        }
        return step;
    }
};

// ---------------- GL helpers ----------------
static unsigned int makeSSBO(size_t bytes, const void* data, int binding) {
    unsigned int b;
    glGenBuffers(1, &b);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, b);
    glBufferData(GL_SHADER_STORAGE_BUFFER, bytes, data, GL_DYNAMIC_DRAW);
    if (binding >= 0) glBindBufferBase(GL_SHADER_STORAGE_BUFFER, binding, b);
    return b;
}

int main(int argc, char** argv) {
    int W      = (argc > 1) ? std::atoi(argv[1]) : (1 << 20);   // lanes (frontier width)
    int L      = (argc > 2) ? std::atoi(argv[2]) : 8;           // layers (steps)
    int trials = (argc > 3) ? std::atoi(argv[3]) : 5;

    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    GLFWwindow* window = glfwCreateWindow(64, 64, "frontier", nullptr, nullptr);
    if (!window) { std::cerr << "window failed\n"; return 1; }
    glfwMakeContextCurrent(window);
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) { std::cerr << "glad failed\n"; return 1; }

    std::vector<Node> nodes;
    std::vector<uint32_t> fanStart, fanList;
    buildCircuit(W, L, nodes, fanStart, fanList);
    const size_t total = nodes.size();                   // W*L + 1

    std::cout << "Renderer: " << glGetString(GL_RENDERER)
              << "\nW = " << W << " lanes, L = " << L << " layers, nodes = " << total
              << ", trials = " << trials << "\n\n";

    // ---------- GPU buffers (frontier lists only need W entries) ----------
    std::vector<uint32_t> zerosTotal(total, 0), zerosW(W + 1, 0);
    std::vector<uint32_t> seed(fanList.begin() + fanStart[0], fanList.begin() + fanStart[1]);
    uint32_t zero = 0;

    makeSSBO(total * sizeof(Node), nodes.data(), 0);
    unsigned int bufValues  = makeSSBO(total * 4, zerosTotal.data(), 1);
    unsigned int bufA       = makeSSBO((W + 1) * 4, zerosW.data(), -1);
    unsigned int bufB       = makeSSBO((W + 1) * 4, zerosW.data(), -1);
    makeSSBO((W + 1) * 4, zerosW.data(), 3);             // pending
    makeSSBO(fanStart.size() * 4, fanStart.data(), 4);
    makeSSBO(fanList.size() * 4, fanList.data(), 5);
    makeSSBO(total * 4, zerosTotal.data(), 6);           // marks
    unsigned int bufCounter = makeSSBO(4, &zero, 8);

    unsigned int evalProg   = compileCompute(EVAL_SRC,   "eval");
    unsigned int commitProg = compileCompute(COMMIT_SRC, "commit");
    int locEvalCount   = glGetUniformLocation(evalProg,   "curCount");
    int locCommitCount = glGetUniformLocation(commitProg, "curCount");
    int locStamp       = glGetUniformLocation(commitProg, "stamp");

    uint32_t gpuStamp = 0;
    uint32_t srcVal = 0;
    std::vector<uint32_t> gpuSizes;

    // One full propagation on the GPU. Returns number of steps.
    auto gpuPropagate = [&](bool record) -> int {
        srcVal ^= 1u;                                    // toggle source (untimed upload of 4 bytes)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, bufValues);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, 4, &srcVal);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, bufA);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, seed.size() * 4, seed.data());

        unsigned int curBuf = bufA, nxtBuf = bufB;
        uint32_t count = (uint32_t)seed.size();
        int steps = 0;

        while (count > 0) {
            ++steps;
            if (record) gpuSizes.push_back(count);

            glBindBuffer(GL_SHADER_STORAGE_BUFFER, bufCounter);
            glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, 4, &zero);

            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, curBuf);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, nxtBuf);
            GLuint groups = (count + 63) / 64;

            glUseProgram(evalProg);                      // PHASE 1
            glUniform1ui(locEvalCount, count);
            glDispatchCompute(groups, 1, 1);
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

            glUseProgram(commitProg);                    // PHASE 2
            glUniform1ui(locCommitCount, count);
            glUniform1ui(locStamp, ++gpuStamp);
            glDispatchCompute(groups, 1, 1);
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

            glBindBuffer(GL_SHADER_STORAGE_BUFFER, bufCounter);
            glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, 4, &count);   // syncs
            std::swap(curBuf, nxtBuf);
        }
        return steps;
    };

    CpuSim cpu(nodes, fanStart, fanList);
    std::vector<uint32_t> cpuSizes;

    // ---------- warm-up (untimed) ----------
    int cpuSteps = cpu.propagate(&cpuSizes);
    int gpuSteps = gpuPropagate(true);
    glFinish();

    // ---------- timed trials ----------
    double cpuTotal = 0, gpuTotal = 0;
    for (int t = 0; t < trials; ++t) {
        auto c0 = Clock::now();
        cpu.propagate();
        cpuTotal += msSince(c0);

        auto g0 = Clock::now();
        gpuPropagate(false);
        glFinish();
        gpuTotal += msSince(g0);
    }
    double cpuMs = cpuTotal / trials, gpuMs = gpuTotal / trials;

    // ---------- verify ----------
    std::vector<uint32_t> gpuVal(total);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, bufValues);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, total * 4, gpuVal.data());

    uint64_t evals = 0;
    for (auto s : gpuSizes) evals += s;

    std::cout << "steps   CPU=" << cpuSteps << "  GPU=" << gpuSteps
              << "\nevals   " << evals << " per propagation (frontier width " << W << ")"
              << "\nvalues  " << (gpuVal == cpu.val ? "MATCH" : "MISMATCH")
              << "\nsizes   " << (gpuSizes == cpuSizes ? "MATCH" : "MISMATCH")
              << "\n\naverage over " << trials << " trials (after warm-up):"
              << "\n  CPU (1 thread) = " << cpuMs << " ms   (" << cpuMs * 1e6 / evals << " ns/eval)"
              << "\n  GPU            = " << gpuMs << " ms   (" << gpuMs * 1e6 / evals << " ns/eval)"
              << "\n  speed-up       = " << cpuMs / gpuMs << "x\n";

    glfwTerminate();
    return 0;
}
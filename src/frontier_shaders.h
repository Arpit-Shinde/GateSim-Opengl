#pragma once
#include <glad/glad.h>
#include <iostream>

// ---------------------------------------------------------------
// PHASE 1: evaluate every frontier node against the SAME committed
// state. Nothing in Values is written here.
// ---------------------------------------------------------------
static const char* EVAL_SRC = R"(#version 430 core
layout(local_size_x = 64) in;

layout(std430, binding = 0) readonly buffer Nodes   { ivec4 nodes[]; };   // x=type, y=in0, z=in1
layout(std430, binding = 1) readonly buffer Values  { uint  values[]; };  // committed state
layout(std430, binding = 2) readonly buffer Cur     { uint  curList[]; }; // current frontier
layout(std430, binding = 3) writeonly buffer Pending{ uint  pending[]; }; // bit0 = new value, bit1 = changed

uniform uint curCount;

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= curCount) return;

    uint id  = curList[i];
    ivec4 nd = nodes[id];

    uint a = (nd.y >= 0) ? values[nd.y] : 0u;   // unconnected pin reads 0
    uint b = (nd.z >= 0) ? values[nd.z] : 0u;

    uint old = values[id];
    uint r = (nd.x == 1) ? (a ^ b) : old;       // type 1 = XOR

    uint changed = (r != old) ? 1u : 0u;
    pending[i] = r | (changed << 1);
}
)";

// ---------------------------------------------------------------
// PHASE 2: commit all results together and build the next frontier.
// ---------------------------------------------------------------
static const char* COMMIT_SRC = R"(#version 430 core
layout(local_size_x = 64) in;

layout(std430, binding = 1) buffer         Values   { uint values[]; };
layout(std430, binding = 2) readonly buffer Cur     { uint curList[]; };
layout(std430, binding = 3) readonly buffer Pending { uint pending[]; };
layout(std430, binding = 4) readonly buffer FanStart{ uint fanStart[]; }; // CSR offsets, size N+1
layout(std430, binding = 5) readonly buffer FanList { uint fanList[]; };  // CSR targets
layout(std430, binding = 6) buffer         Marks    { uint marks[]; };    // per-node stamp
layout(std430, binding = 7) writeonly buffer Nxt    { uint nxtList[]; };  // next frontier
layout(std430, binding = 8) buffer         Counter  { uint nxtCount; };

uniform uint curCount;
uniform uint stamp;

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= curCount) return;

    uint p = pending[i];
    if ((p & 2u) == 0u) return;          // unchanged: does not activate neighbours

    uint id = curList[i];
    values[id] = p & 1u;                 // commit

    for (uint k = fanStart[id]; k < fanStart[id + 1u]; ++k) {
        uint o = fanList[k];
        if (atomicExchange(marks[o], stamp) != stamp) {   // first thread to claim o this step
            uint slot = atomicAdd(nxtCount, 1u);
            nxtList[slot] = o;
        }
    }
}
)";

inline unsigned int compileCompute(const char* src, const char* name) {
    unsigned int sh = glCreateShader(GL_COMPUTE_SHADER);
    glShaderSource(sh, 1, &src, nullptr);
    glCompileShader(sh);

    int ok = 0;
    glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[2048];
        glGetShaderInfoLog(sh, sizeof(log), nullptr, log);
        std::cerr << name << " compile error:\n" << log << "\n";
        std::exit(1);
    }

    unsigned int prog = glCreateProgram();
    glAttachShader(prog, sh);
    glLinkProgram(prog);
    glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[2048];
        glGetProgramInfoLog(prog, sizeof(log), nullptr, log);
        std::cerr << name << " link error:\n" << log << "\n";
        std::exit(1);
    }
    glDeleteShader(sh);
    return prog;
}
export const MAX_THREAD_NUM = 1024;

export const shader1 = `#version 310 es
layout (local_size_x = ${MAX_THREAD_NUM}, local_size_y = 1, local_size_z = 1) in;
layout (std430, binding = 0) buffer SSBO {
  float data[];
} ssbo;
shared float sharedData[${MAX_THREAD_NUM}];

void main() {
  sharedData[gl_LocalInvocationID.x] = ssbo.data[gl_GlobalInvocationID.x];
  memoryBarrierShared();
  barrier();
  
  uint offset = gl_WorkGroupID.x * gl_WorkGroupSize.x;
  
  float tmp;
  for (uint k = 2u; k <= gl_WorkGroupSize.x; k <<= 1) {
    for (uint j = k >> 1; j > 0u; j >>= 1) {
      uint ixj = (gl_GlobalInvocationID.x ^ j) - offset;
      if (ixj > gl_LocalInvocationID.x) {
        if ((gl_GlobalInvocationID.x & k) == 0u) {
          if (sharedData[gl_LocalInvocationID.x] > sharedData[ixj]) {
            tmp = sharedData[gl_LocalInvocationID.x];
            sharedData[gl_LocalInvocationID.x] = sharedData[ixj];
            sharedData[ixj] = tmp;
          }
        }
        else
        {
          if (sharedData[gl_LocalInvocationID.x] < sharedData[ixj]) {
            tmp = sharedData[gl_LocalInvocationID.x];
            sharedData[gl_LocalInvocationID.x] = sharedData[ixj];
            sharedData[ixj] = tmp;
          }
        }
      }
      memoryBarrierShared();
      barrier();
    }
  }
  ssbo.data[gl_GlobalInvocationID.x] = sharedData[gl_LocalInvocationID.x];
}
`;

export const shader2 = `#version 310 es
layout (local_size_x = ${MAX_THREAD_NUM}, local_size_y = 1, local_size_z = 1) in;

layout (binding = 0) uniform tonic {
    uvec4 numElements;
};
layout (std430, binding = 1) buffer SSBO {
    float data[];
  } ssbo;
void main() {
   float tmp;
  uint ixj = gl_GlobalInvocationID.x ^ numElements.y;
  if (ixj > gl_GlobalInvocationID.x)
  {
    if ((gl_GlobalInvocationID.x & numElements.x) == 0u)
    {
      if (ssbo.data[gl_GlobalInvocationID.x] > ssbo.data[ixj])
      {
        tmp = ssbo.data[gl_GlobalInvocationID.x];
        ssbo.data[gl_GlobalInvocationID.x] = ssbo.data[ixj];
        ssbo.data[ixj] = tmp;
      }
    }
    else
    {
      if (ssbo.data[gl_GlobalInvocationID.x] < ssbo.data[ixj])
      {
        tmp = ssbo.data[gl_GlobalInvocationID.x];
        ssbo.data[gl_GlobalInvocationID.x] = ssbo.data[ixj];
        ssbo.data[ixj] = tmp;
      }
    }
  }
}
`;
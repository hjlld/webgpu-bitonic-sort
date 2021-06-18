import { MAX_THREAD_NUM, shader1, shader2 } from './GLSLComputeShaderCode';
import glslangModule, { Glslang } from '@webgpu/glslang/dist/web-devel-onefile/glslang';

export class WebGPUSort {

    public adapter?: GPUAdapter | null;

    public device?: GPUDevice;

    public glslang?: Glslang;

    constructor() {

        if (!navigator.gpu) {

            throw new Error('WebGPU not supported!');

        }

    }

    public async Init() {

        this.adapter = await navigator.gpu.requestAdapter({

            powerPreference: 'high-performance'

        });

        if (!this.adapter) {

            throw new Error('Adapter init failed!');

        }

        this.device = await this.adapter.requestDevice();

        console.time('Init GLSLang');

        this.glslang = await glslangModule();

        console.timeEnd('Init GLSLang');

    }

    public Validate( array: Float32Array ) {

        let length = array.length;

        for (let i = 0; i < length; i++) {

            if (i !== length - 1 && array[i] > array[i + 1]) {

                console.error('validation error:', i, array[i], array[i + 1]);

                return false;

            }
        }

        return true;

    }

    public async Run(array: Float32Array): Promise<Float32Array> {

        if (!this.device || !this.glslang) {

            throw new Error('Device not found or GLSL compiler not ready!');

        }

        let commandEncoder = this.device.createCommandEncoder();

        let passEncoder = commandEncoder.beginComputePass();

        let length = array.length;

        let size = array.byteLength;

        let threadgroupsPerGrid = Math.max(1, length / MAX_THREAD_NUM);

        let inputBuffer = this.device!.createBuffer({

            mappedAtCreation: true,
            size: size,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST

        });

        let inputRange = inputBuffer.getMappedRange();

        new Float32Array(inputRange).set(array);

        inputBuffer.unmap();

        let shaderModule1 = this.device.createShaderModule({

            label: 'shader1',

            code: this.glslang.compileGLSL(shader1, 'compute', false)

        });

        let bindGroupLayout1 = this.device.createBindGroupLayout({

            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: 'storage'
                    }
                }
            ]

        });

        let pipelineLayout1 = this.device.createPipelineLayout({

            bindGroupLayouts: [ bindGroupLayout1 ]

        });
        
        let pipeline1 = await this.device.createComputePipelineAsync({

            label: 'pipeline1',

            compute: {

                module: shaderModule1,

                entryPoint: 'main'

            },

            layout: pipelineLayout1

        });

        let shaderModule2 = this.device.createShaderModule({

            label: 'shader2',

            code: this.glslang.compileGLSL(shader2, 'compute', false)

        });
        
        let bindGroupLayout2 = this.device.createBindGroupLayout({

            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: 'uniform'
                    }
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: 'storage'
                    }
                }
            ]

        });

        let pipelineLayout2 = this.device.createPipelineLayout({

            bindGroupLayouts: [ bindGroupLayout2 ]

        });
        
        let pipeline2 = await this.device.createComputePipelineAsync({

            label: 'pipeline2',

            compute: {

                module: shaderModule2,

                entryPoint: 'main'

            },

            layout: pipelineLayout2

        });

        let bindGroup1 = this.device.createBindGroup({

            layout: bindGroupLayout1,

            entries: [

                {
                    binding: 0,
                    resource: {
                        buffer: inputBuffer
                    }
                }

            ]

        });

        passEncoder.setPipeline(pipeline1);

        passEncoder.setBindGroup(0, bindGroup1);

        passEncoder.dispatch(threadgroupsPerGrid, 1, 1);

        let uniformBuffer = this.device.createBuffer({

            size: 16,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST

        });

        let bindGroup2 = this.device.createBindGroup({

            layout: bindGroupLayout2,

            entries: [
                
                {
                    binding: 0,
                    resource: {
                        buffer: uniformBuffer
                    }
                },

                {
                    binding: 1,
                    resource: {
                        buffer: inputBuffer
                    }
                }

            ]

        });

        if (threadgroupsPerGrid > 1) {

            passEncoder.setPipeline(pipeline2);

            passEncoder.setBindGroup(0, bindGroup2);

            for (let k = threadgroupsPerGrid; k <= length; k <<= 1) {

                for (let j = k >> 1; j > 0; j >>= 1) {
            
                    let uniform = new Uint32Array([k, j, 0, 0]);
                    
                    this.device.queue.writeBuffer(uniformBuffer, 0, uniform);

                    passEncoder.dispatch(threadgroupsPerGrid, 1, 1);
                    
                }

            }

        }

        passEncoder.endPass();

        let resultBufferToRead = this.device.createBuffer({

            size,

            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ

        });

        commandEncoder.copyBufferToBuffer(inputBuffer, 0, resultBufferToRead, 0, size);

        this.device.queue.submit( [ commandEncoder.finish() ] );

        await resultBufferToRead.mapAsync(GPUMapMode.READ);

        let resultMappedRange: ArrayBuffer = resultBufferToRead.getMappedRange();

        let result = new Float32Array(resultMappedRange);

        return result;

    }

}
import { shader1, shader2 } from './ComputeShaderCode.wgsl';

export class WebGPUSort {

    public adapter?: GPUAdapter | null;

    public device?: GPUDevice;

    public maxThreadNum: number = 256;

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

        this.maxThreadNum = this.device.limits.maxComputeWorkgroupSizeX;

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

        if (!this.device) {

            throw new Error('Device not found!');

        }

        let length = array.length;

        let byteLength = array.byteLength;

        let threadgroupsPerGrid = Math.max(1, length / this.maxThreadNum);

        let offset = Math.log2( length ) - ( Math.log2( this.maxThreadNum * 2 + 1) );

        let inputBuffer = this.device!.createBuffer({

            label: 'input',
            mappedAtCreation: true,
            size: byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST

        });

        let inputRange = inputBuffer.getMappedRange();

        new Float32Array(inputRange).set(array);

        inputBuffer.unmap();

        let shaderModule1 = this.device.createShaderModule({

            label: 'shader1',
            code: shader1( this.maxThreadNum )

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

        let commandEncoder = this.device.createCommandEncoder();

        let passEncoder = commandEncoder.beginComputePass();

        passEncoder.setPipeline(pipeline1);

        passEncoder.setBindGroup(0, bindGroup1);

        passEncoder.dispatch(threadgroupsPerGrid, 1, 1);

        passEncoder.endPass();

        this.device.queue.submit( [ commandEncoder.finish() ] );

        let uniform = new Uint32Array([0, 0, 0, 0]);

        let uniformBuffer = this.device.createBuffer({

            size: 16,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST

        });

        
        let shaderModule2 = this.device.createShaderModule({

            label: 'shader2',
            code: shader2( this.maxThreadNum )

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

            for (let k = threadgroupsPerGrid >> offset; k <= length; k = k << 1) {

                for (let j = k >> 1; j > 0; j = j >> 1) {

                    let commandEncoder2 = this.device.createCommandEncoder();

                    let passEncoder2 = commandEncoder2.beginComputePass();
            
                    passEncoder2.setPipeline(pipeline2);

                    passEncoder2.setBindGroup(0, bindGroup2);

                    uniform[ 0 ] = k;

                    uniform[ 1 ] = j;
                    
                    this.device.queue.writeBuffer(uniformBuffer, 0, uniform);

                    passEncoder2.dispatch(threadgroupsPerGrid, 1, 1);

                    passEncoder2.endPass();

                    this.device.queue.submit( [ commandEncoder2.finish() ] );

                }

            }

        }

        let lastCommandEncoder = this.device.createCommandEncoder();

        let resultBufferToRead = this.device.createBuffer({

            size: byteLength,

            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ

        });

        lastCommandEncoder.copyBufferToBuffer(inputBuffer, 0, resultBufferToRead, 0, byteLength);

        this.device.queue.submit( [ lastCommandEncoder.finish() ] );

        console.time( 'GPU sort - result buffer map async')

        await resultBufferToRead.mapAsync(GPUMapMode.READ);

        console.timeEnd( 'GPU sort - result buffer map async')

        let resultMappedRange: ArrayBuffer = resultBufferToRead.getMappedRange();

        let result = new Float32Array(resultMappedRange);

        return result;

    }

    public Dispose() {

        this.device?.destroy();

    }



}
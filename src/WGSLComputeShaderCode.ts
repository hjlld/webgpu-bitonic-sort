export const MAX_THREAD_NUM = 1024;
export const MAX_GROUP_NUM = 2048;

export const shader1 = `
struct structFixedData {
    data: array<f32, ${MAX_THREAD_NUM}>;
};

[[block]]
struct ssbo {
    data: array<f32>;
};

[[group(0), binding(0)]]
var<storage, write> input: ssbo;

var<workgroup> sharedData: structFixedData;

[[stage(compute), workgroup_size(${MAX_THREAD_NUM},1,1)]]
fn main(
    [[builtin(local_invocation_id)]] local_id: vec3<u32>,
    [[builtin(global_invocation_id)]] global_id: vec3<u32>,
    [[builtin(workgroup_id)]] group_id: vec3<u32>,
) {
    let localIdX: u32 = local_id.x;
    let globalIdX: u32 = global_id.x;

    sharedData.data[ localIdX ] = input.data[ globalIdX ];

    workgroupBarrier();
    storageBarrier();

    let offset: u32 = group_id.x * ${MAX_THREAD_NUM}u;

    var tmp: f32;

    for ( var k: u32 = 2u; k <= ${MAX_THREAD_NUM}u; k = k << 1u ) {

        for ( var j: u32 = k >> 1u; j > 0u; j = j >> 1u ) {

            let ixj: u32 = ( globalIdX ^ j ) - offset;

            if ( ixj > localIdX ) {

                if ( ( globalIdX & k ) == 0u ) {

                    if ( sharedData.data[ localIdX ] > sharedData.data[ ixj ] ) {

                        tmp = sharedData.data[ localIdX ];

                        sharedData.data[ localIdX ] = sharedData.data[ ixj ];

                        sharedData.data[ ixj ] = tmp;

                    }

                } else {

                    if ( sharedData.data[ localIdX ] < sharedData.data[ ixj ] ) {

                        tmp = sharedData.data[ localIdX ];
                        
                        sharedData.data[ localIdX ] = sharedData.data[ ixj ];

                        sharedData.data[ ixj ] = tmp;

                    }

                }

            }

            workgroupBarrier();
            storageBarrier();
            
        }

    }

    input.data[ globalIdX ] = sharedData.data[ localIdX ];

}
`;

export const shader2 = `
[[block]]
struct ssbo {
    data: array<f32>;
};

[[block]]
struct struct_Uniform {
    data: vec4<u32>;
};

[[group(0), binding(0)]]
var<uniform> tonic: struct_Uniform;

[[group(0), binding(1)]]
var<storage, write> input: ssbo;

[[stage(compute), workgroup_size(${MAX_THREAD_NUM},1,1)]]
fn main( [[builtin(local_invocation_id)]] global_id: vec3<u32> ) {

    let globalIdX: u32 = global_id.x;

    var tmp: f32;

    let ixj: u32 = globalIdX ^ tonic.data.y;

    if ( ixj > globalIdX ) {

        if ( ( globalIdX & tonic.data.x ) == 0u ) {

            if ( input.data[ globalIdX ] > input.data[ ixj ] ) {

                tmp = input.data[ globalIdX ];

                input.data[ globalIdX ] = input.data[ ixj ];

                input.data[ ixj ] = tmp;

            }

        } else {

            if ( input.data[ globalIdX ] < input.data[ ixj ] ) {

                tmp = input.data[ globalIdX ];

                input.data[ globalIdX ] = input.data[ ixj ];

                input.data[ ixj ] = tmp;

            }

        }

    }

}
`;
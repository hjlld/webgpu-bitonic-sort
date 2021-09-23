import './style.css'
import { WebGPUSort } from './WebGPUSort';

let main = async () => {

    let exponent = 23;

    let array = new Float32Array( Math.pow( 2, exponent ) );

    for ( let i = 0; i < array.length; ++ i ) {

        array[ i ] = Math.random();

    }

    let sort = new WebGPUSort();

    await sort.Init();

    console.time( 'GPU sort' );

    let gpuSortResult = await sort.Run( array );

    console.timeEnd( 'GPU sort' );

    console.log( gpuSortResult );

    console.time( 'CPU sort' );

    if ( sort.Validate( gpuSortResult ) ) {

        console.log( 'GPU sort validation passed!' )

    };

    let cpuSortResult = array.slice().sort( ( a, b ) => a - b );

    console.timeEnd( 'CPU sort' );

    console.log( cpuSortResult );

  
}

window.addEventListener( 'DOMContentLoaded', main );
#include "sptp_exp_opt.hpp"
#include <cmath>

#define WARPSIZE 32 
#define MAX_IN1_IR_CNT 32 // all parity up to l_max 5 < 32 (12) x 32 = 384B
#define MAX_NUM_PATH 256 // 6^3 = 216 < 256 x (4+4+4+4+4 = 20) 5120B
#define MAX_U_FIBER_CNT 8192 // up to L_max =5 (4B) 
#define MAX_U_CG_VAL_CNT 256 // (4B) => 1024B

__constant__ int in1_idxing[MAX_IN1_IR_CNT];
__constant__ int in1_ival[MAX_IN1_IR_CNT];
__constant__ int in1_related_path_idx[MAX_IN1_IR_CNT];

__constant__ uint path_array1[MAX_NUM_PATH];
__constant__ uchar4 path_array2[MAX_NUM_PATH];
__constant__ ushort2 per_path_fiber_start[MAX_NUM_PATH];
__constant__ float path_weight[MAX_NUM_PATH];
__constant__ int per_path_weight_pos[MAX_NUM_PATH];

__constant__ uchar4 fiber_array[MAX_U_FIBER_CNT];
__constant__ float unique_cg_val[MAX_U_CG_VAL_CNT];

template <typename scalar_t>
__global__ void triple_bwd_sptp_lienar_kernel_v2_shared_exp(
    const float* __restrict__ mem_dT_dGin1,  // tx, from grad_output
    const float* __restrict__ mem_dT_dGin2,  // ty, from grad_output
    const float* __restrict__ mem_dT_dGW,    // t_theta, from grad_output
    const float* __restrict__ mem_dT_dGLo,   // tL, from grad_output
    
    const float* __restrict__ mem_dL_dO,  // Lo, from ctx
    const float* __restrict__ mem_dG_dLx, // gx, from ctx
    const float* __restrict__ mem_dG_dLy, // gy, from ctx
    const float* __restrict__ mem_dG_dLW, // g_theta, from ctx

    const float* __restrict__ in1,
    const float* __restrict__ in2,
    const float* __restrict__ weight,
    
    const int* __restrict__ per_edge_src,
    const int* __restrict__ per_edge_dst,

    float* __restrict__ mem_dT_din1,
    float* __restrict__ mem_dT_din2,
    float* __restrict__ mem_dT_dW,
    float* __restrict__ mem_dT_dLo,
    float* __restrict__ mem_dT_dgx,
    float* __restrict__ mem_dT_dgy,
    float* __restrict__ mem_dT_dgW,
    float* __restrict__ mem_debug,
    const ushort* __restrict__ per_exec_info,

    const size_t batch_size,
    const size_t out_size,
    const size_t weight_size,
    const size_t in1_size,
    const size_t in2_size,
    const size_t perwarp_in2_size,
    const size_t path_cnt,
    const size_t max_ir_dim
    )
    {
    extern __shared__ scalar_t shmem[];
    // Input dL_dO => batch, ir, mul order
    // 2D grid, 2D block
    // grid (path, batch), block (mul(same path), batch)
    // intra-warp (u parallel) x , inter-warp (batch) y 
    const int global_t_batch_idx = blockIdx.x * blockDim.y + threadIdx.y;

    if(global_t_batch_idx >= batch_size) return;

    const int src_idx = per_edge_src[global_t_batch_idx];
    const int dst_idx = per_edge_dst[global_t_batch_idx];

    // check given path (path per thread_block)

    // start_end of out for a block
    // divide by u 

    // load all in2 to shmem
    // load all nnz fiber to shmem
    // load cg value (to register?)
    // load all w 
    // sync
    
    // no init needed just copy

    scalar_t* my_batch_shmem_start = shmem + threadIdx.y * (blockDim.x * (max_ir_dim*12+perwarp_in2_size*2) );

    scalar_t* my_shmem_in1 = my_batch_shmem_start + threadIdx.x*max_ir_dim;
    scalar_t* my_shmem_dT_dGin1 = my_batch_shmem_start + blockDim.x*(max_ir_dim) + threadIdx.x*max_ir_dim; 
    scalar_t* my_shmem_dT_dGLo = my_batch_shmem_start + blockDim.x*(max_ir_dim*2) + threadIdx.x*max_ir_dim; 

    scalar_t* my_shmem_T_uvuv = my_batch_shmem_start +  blockDim.x*(max_ir_dim*3) + threadIdx.x*max_ir_dim;
    scalar_t* my_shmem_TF_uvuv = my_batch_shmem_start +  blockDim.x*(max_ir_dim*4) + threadIdx.x*max_ir_dim;
    scalar_t* my_shmem_F_uvuv = my_batch_shmem_start +  blockDim.x*(max_ir_dim*5) + threadIdx.x*max_ir_dim;
    scalar_t* my_shmem_uvuv = my_batch_shmem_start +  blockDim.x*(max_ir_dim*6) + threadIdx.x*max_ir_dim;
    
    scalar_t* my_shmem_dL_dO = my_batch_shmem_start + blockDim.x*(max_ir_dim*7) + threadIdx.x*max_ir_dim;
    scalar_t* my_shmem_dG_dLx = my_batch_shmem_start + blockDim.x*(max_ir_dim*8) + threadIdx.x*max_ir_dim;
   
    scalar_t* my_shmem_dT_din1 = my_batch_shmem_start + blockDim.x*(max_ir_dim*9) + threadIdx.x*max_ir_dim;
    scalar_t* my_shmem_dT_dgx = my_batch_shmem_start + blockDim.x*(max_ir_dim*10) + threadIdx.x*max_ir_dim;

    scalar_t* my_shmem_dT_din2 = my_batch_shmem_start + blockDim.x*(max_ir_dim*11) + threadIdx.x*perwarp_in2_size;
    scalar_t* my_shmem_dT_dgy = my_batch_shmem_start + blockDim.x*(max_ir_dim*11+perwarp_in2_size) + threadIdx.x*perwarp_in2_size;
    
    scalar_t* shmem_scratch = my_batch_shmem_start + blockDim.x*(max_ir_dim*11+perwarp_in2_size*2); // first save dT_dLo

    scalar_t* shmem_in2 = shmem + blockDim.y * (blockDim.x * (max_ir_dim*12+perwarp_in2_size*2)) + threadIdx.y * in2_size;
    scalar_t* shmem_dG_dLy = shmem + blockDim.y * (blockDim.x * (max_ir_dim*12+perwarp_in2_size*2)) + blockDim.y*(in2_size) + threadIdx.y * in2_size;
    scalar_t* shmem_dT_dGin2 = shmem + blockDim.y * (blockDim.x * (max_ir_dim*12+perwarp_in2_size*2)) + (blockDim.y*in2_size*2) + threadIdx.y * in2_size;


    // dL_dO size : WARPSIZE * MAX_IR (all warps) * concurrent_batch (warp cnt)
    // dL_dO size : out_size 
    // (which i_in1 path, batch) (mul, batch)
    // need a lot of register.. (unless i make macro for all cases)

    // what defines the target_in1 ?? that is the question need z axis?

    // load part of in1 from main mem
    // in1 (z, mul, ir)
    const int exec_idx = blockIdx.y;
    // look up a array ushort2 [target_in1, channel_chunk]
    ushort2* per_exec_info_pkt = (ushort2*) per_exec_info;
    ushort2 exec_info = per_exec_info_pkt[exec_idx];
    const int target_in1 = exec_info.x;
    const int channel_chunk_idx = exec_info.y;

    const int i_val = in1_ival[target_in1];
    const int in1_start = in1_idxing[target_in1] + i_val * WARPSIZE * channel_chunk_idx;
    const int in1_end = in1_start + i_val * WARPSIZE;
    const int path_idx_start = in1_related_path_idx[target_in1];
    const int path_idx_end = in1_related_path_idx[target_in1+1];

    // using reg_dL_din1 for dummy => need to initialize ...
    unsigned long long in1_idx = src_idx*in1_size + in1_start+threadIdx.x;
    for(int shmem_idx = threadIdx.x; in1_idx < src_idx*in1_size + in1_end; shmem_idx+=WARPSIZE, in1_idx+=WARPSIZE) {
        shmem_scratch[shmem_idx] = in1[in1_idx];
    }
    __syncwarp();
    for(int i =0, shmem_idx = threadIdx.x*i_val; i<i_val; i++, shmem_idx++){
        my_shmem_in1[i] = shmem_scratch[shmem_idx];
    }
    __syncwarp();
    
    in1_idx = src_idx*in1_size + in1_start+threadIdx.x;
    for(int shmem_idx = threadIdx.x; in1_idx < src_idx*in1_size + in1_end; shmem_idx+=WARPSIZE, in1_idx+=WARPSIZE) {
        shmem_scratch[shmem_idx] = mem_dT_dGin1[in1_idx];
    }
    __syncwarp();
    for(int i =0, shmem_idx = threadIdx.x*i_val; i<i_val; i++, shmem_idx++){
        my_shmem_dT_dGin1[i] = shmem_scratch[shmem_idx];
    }
    __syncwarp();
    
    in1_idx = src_idx*in1_size + in1_start+threadIdx.x;
    for(int shmem_idx = threadIdx.x; in1_idx < src_idx*in1_size + in1_end; shmem_idx+=WARPSIZE, in1_idx+=WARPSIZE) {
        shmem_scratch[shmem_idx] = mem_dG_dLx[in1_idx];
    }
    __syncwarp();
    for(int i =0, shmem_idx = threadIdx.x*i_val; i<i_val; i++, shmem_idx++){
        my_shmem_dG_dLx[i] = shmem_scratch[shmem_idx];
    }

    // load all in2 from main mem
    unsigned long long in2_idx = global_t_batch_idx*in2_size + threadIdx.x;
    for(int shmem_idx = threadIdx.x; shmem_idx < in2_size; in2_idx+=WARPSIZE, shmem_idx+=WARPSIZE) {
        shmem_in2[shmem_idx] = in2[in2_idx];
        shmem_dG_dLy[shmem_idx] = mem_dG_dLy[in2_idx];
        shmem_dT_dGin2[shmem_idx] = mem_dT_dGin2[in2_idx];
    }
    __syncwarp();
    
    for(int i=0; i<max_ir_dim;i++){
        my_shmem_dT_din1[i] = 0.0;
        my_shmem_dT_dgx[i] = 0.0;
    }
    for(int i=0; i<in2_size;i++){
        my_shmem_dT_din2[i] = 0.0;
        my_shmem_dT_dgy[i] = 0.0;
    }

    // for path_chunk
    // path idx == k idx
    // path index == 
    const unsigned long long g_t_dldo_start = dst_idx*out_size;
    // const unsigned long long g_t_dfdo_start = global_t_batch_idx*out_size;
    const unsigned long long g_t_dfdo_start = dst_idx*out_size;
    const unsigned long long g_t_w_start = global_t_batch_idx*weight_size;

    for(int path_idx=path_idx_start; path_idx < path_idx_end; path_idx++){
        const uint path_info1 = path_array1[path_idx]; // k_start
        const uchar4 path_info2 = path_array2[path_idx]; // k_val, j_start, j_val, j_end

        for(int i=0; i<max_ir_dim;i++){
            my_shmem_uvuv[i] = 0.0;
            my_shmem_F_uvuv[i] = 0.0;
            my_shmem_TF_uvuv[i] = 0.0;
            my_shmem_T_uvuv[i] = 0.0;
        }
        
        // stall due to global memory access (better if it is load to shared memory and accessed)
        // possible optimization point with gather scatter
        const unsigned long long out_start = g_t_dldo_start + path_info1 + path_info2.x*WARPSIZE*channel_chunk_idx; 
        const unsigned long long out_end = out_start + path_info2.x*WARPSIZE; 
        unsigned long long out_idx = out_start + threadIdx.x; 

        for(int shmem_idx = threadIdx.x; out_idx < out_end; out_idx+=WARPSIZE, shmem_idx+=WARPSIZE){
            shmem_scratch[shmem_idx] = mem_dL_dO[out_idx];
        }
        __syncwarp();
      
        // load k_val amount from shmem
        for(int i =0, shmem_idx = threadIdx.x*path_info2.x; i<path_info2.x; i++, shmem_idx++){
            my_shmem_dL_dO[i] = shmem_scratch[shmem_idx];
        }

        __syncwarp();
        out_idx = out_start + threadIdx.x; 
        for(int shmem_idx = threadIdx.x; out_idx < out_end; out_idx+=WARPSIZE, shmem_idx+=WARPSIZE){
            shmem_scratch[shmem_idx] = mem_dT_dGLo[out_idx];
        }
        __syncwarp();
        // load k_val amount from shmem
        for(int i =0, shmem_idx = threadIdx.x*path_info2.x; i<path_info2.x; i++, shmem_idx++){
            my_shmem_dT_dGLo[i] = shmem_scratch[shmem_idx];
        }

        // odd number of k_val (2n+1) no bank conflict
        
        // Loading Weight from global memory is a major memory bottleneck
        const unsigned long long weight_pos = g_t_w_start + per_path_weight_pos[path_idx]+ WARPSIZE*channel_chunk_idx + threadIdx.x;
        float reg_w_path_norm = weight[weight_pos] * path_weight[path_idx];  // w pw
        float reg_F_w_path_norm = mem_dG_dLW[weight_pos] * path_weight[path_idx];  // gw pw
        float reg_T_w_path_norm = mem_dT_dGW[weight_pos] * path_weight[path_idx];  // tw pw
        
        const ushort2 fiber_idx_info = per_path_fiber_start[path_idx];
        // for nnz in the fiber
        // uchar4 fiber;
        for(ushort fiber_idx = fiber_idx_info.x; fiber_idx < fiber_idx_info.y; fiber_idx++){
            // mult k with all w => dL_duvuv
            uchar4 fiber = fiber_array[fiber_idx]; // i, j, k, cg idx

            //fwd_uvuv
            my_shmem_uvuv[fiber.z] += my_shmem_in1[fiber.x] * shmem_in2[path_info2.y+fiber.y] * unique_cg_val[fiber.w];
            
            //dF_duvuv
            my_shmem_F_uvuv[fiber.z] += (my_shmem_dG_dLx[fiber.x] * shmem_in2[path_info2.y+fiber.y] + my_shmem_in1[fiber.x] * shmem_dG_dLy[path_info2.y+fiber.y]) * unique_cg_val[fiber.w];

            //TF_uvuv
            my_shmem_TF_uvuv[fiber.z] += (my_shmem_dG_dLx[fiber.x] * shmem_dT_dGin2[path_info2.y+fiber.y] + my_shmem_dT_dGin1[fiber.x] * shmem_dG_dLy[path_info2.y+fiber.y]) * unique_cg_val[fiber.w];

            //T_duvuv
            my_shmem_T_uvuv[fiber.z] += (my_shmem_dT_dGin1[fiber.x] * shmem_in2[path_info2.y+fiber.y] + my_shmem_in1[fiber.x] * shmem_dT_dGin2[path_info2.y+fiber.y]) * unique_cg_val[fiber.w];

            float common_dE_dO_deriv = my_shmem_dL_dO[fiber.z] * unique_cg_val[fiber.w];
            float dL_dOuter = common_dE_dO_deriv * reg_F_w_path_norm;  // bijk
            float dE_dOuter = common_dE_dO_deriv * reg_w_path_norm;    // aijk
            float dT_dOuter = common_dE_dO_deriv * reg_T_w_path_norm;  // cijk

            float common_dT_dGLo_deriv = my_shmem_dT_dGLo[fiber.z] * unique_cg_val[fiber.w];
            float dT_dGLo_W = common_dT_dGLo_deriv * reg_w_path_norm;   // dijk
            float dT_dGLo_gW = common_dT_dGLo_deriv * reg_F_w_path_norm;   // eijk
            float dT_dOuter_plus_dT_dGLo_W = dT_dOuter + dT_dGLo_W;     // cijk + dijk

            my_shmem_dT_din1[fiber.x] += dL_dOuter * shmem_dT_dGin2[path_info2.y+fiber.y] + (dT_dOuter_plus_dT_dGLo_W) * shmem_dG_dLy[path_info2.y+fiber.y] + dT_dGLo_gW * shmem_in2[path_info2.y+fiber.y];
            my_shmem_dT_din2[path_info2.y+fiber.y] += dL_dOuter * my_shmem_dT_dGin1[fiber.x]  + (dT_dOuter_plus_dT_dGLo_W) * my_shmem_dG_dLx[fiber.x] + dT_dGLo_gW * my_shmem_in1[fiber.x];
            my_shmem_dT_dgx[fiber.x] += dE_dOuter * shmem_dT_dGin2[path_info2.y+fiber.y] + (dT_dOuter_plus_dT_dGLo_W) * shmem_in2[path_info2.y+fiber.y];
            my_shmem_dT_dgy[path_info2.y+fiber.y] += dE_dOuter * my_shmem_dT_dGin1[fiber.x] + (dT_dOuter_plus_dT_dGLo_W) * my_shmem_in1[fiber.x];
        }

        // debugging
        // for (int i =0, dldo_idx = g_t_dldo_start + path_info1.x + threadIdx.x*path_info2.x; i<path_info2.x; i++, dldo_idx++){
        //     mem_debug[dldo_idx] = my_shmem_uvuv[i];
        // }

        // mem_dL_dW
        float reg_dT_dW = 0.0;
        float reg_dT_dgW = 0.0;
        for(int k_idx = 0; k_idx<path_info2.x; k_idx++){
            reg_dT_dW += (my_shmem_dL_dO[k_idx] * my_shmem_TF_uvuv[k_idx] + my_shmem_dT_dGLo[k_idx] * my_shmem_F_uvuv[k_idx]) * path_weight[path_idx];
            reg_dT_dgW += (my_shmem_dL_dO[k_idx] * my_shmem_T_uvuv[k_idx] + my_shmem_dT_dGLo[k_idx] * my_shmem_uvuv[k_idx]) * path_weight[path_idx];
        }
        mem_dT_dW[weight_pos] = reg_dT_dW;
        mem_dT_dgW[weight_pos] = reg_dT_dgW;

        // dF_dO
        // store out first in shared mem
        for(int i =0, shmem_idx = threadIdx.x*path_info2.x; i<path_info2.x; i++, shmem_idx++){
            shmem_scratch[shmem_idx] = my_shmem_TF_uvuv[i] * reg_w_path_norm + my_shmem_T_uvuv[i] * reg_F_w_path_norm + my_shmem_F_uvuv[i]*reg_T_w_path_norm;
        }
        __syncwarp();
        // store out in main mem
        // unsigned long long df_do_idx = g_t_dfdo_start+path_info1.x + threadIdx.x;
        // for(int shmem_idx = threadIdx.x; df_do_idx< g_t_dfdo_start+ path_info1.y; df_do_idx+=WARPSIZE, shmem_idx+=WARPSIZE) {
        //     mem_dF_dO[df_do_idx] = shmem_scratch[shmem_idx];
        // }
        // use atomic add
        // for(int shmem_idx = threadIdx.x; out_idx < out_end; out_idx+=WARPSIZE, shmem_idx+=WARPSIZE){

        out_idx = out_start + threadIdx.x;
        for(int shmem_idx = threadIdx.x; out_idx < out_end; out_idx+=WARPSIZE, shmem_idx+=WARPSIZE) {
            atomicAdd(mem_dT_dLo+out_idx, shmem_scratch[shmem_idx]);
            // mem_dF_dO[df_do_idx] = shmem_scratch[shmem_idx];
        }
    }
    
    for(int i =0, shmem_idx = threadIdx.x*i_val; i<i_val; i++, shmem_idx++){
        shmem_scratch[shmem_idx] = my_shmem_dT_din1[i];
    }
    __syncwarp();
    // store dL_dA in main mem
    
   in1_idx = src_idx*in1_size + in1_start+threadIdx.x;
    for(int shmem_idx = threadIdx.x; in1_idx < src_idx*in1_size + in1_end; shmem_idx+=WARPSIZE, in1_idx+=WARPSIZE) {
        atomicAdd(mem_dT_din1+in1_idx, shmem_scratch[shmem_idx]);
    }

    __syncwarp();
    for(int i =0, shmem_idx = threadIdx.x*i_val; i<i_val; i++, shmem_idx++){
        shmem_scratch[shmem_idx] = my_shmem_dT_dgx[i];
    }
    __syncwarp();
    // store dL_dA in main mem
    
   in1_idx = src_idx*in1_size + in1_start+threadIdx.x;
    for(int shmem_idx = threadIdx.x; in1_idx < src_idx*in1_size + in1_end; shmem_idx+=WARPSIZE, in1_idx+=WARPSIZE) {
        atomicAdd(mem_dT_dgx+in1_idx, shmem_scratch[shmem_idx]);
    }

    // warp shuffle reduce
    for (int i = 0; i < in2_size; i++) {
        float sum_dT_din2 = my_shmem_dT_din2[i];
        float sum_dT_dgy = my_shmem_dT_dgy[i];
        for (int offset = 1; offset < WARPSIZE; offset *= 2) {
            sum_dT_din2 += __shfl_xor_sync(0xFFFFFFFF, sum_dT_din2, offset);
            sum_dT_dgy += __shfl_xor_sync(0xFFFFFFFF, sum_dT_dgy, offset);
        }
        my_shmem_dT_din2[i] = sum_dT_din2;
        my_shmem_dT_dgy[i] = sum_dT_dgy;
    }

    // load all in2 from main mem 
    // larger by the number of in1 accumulate (len(i_in1))
    const unsigned long long g_dT_din2 = global_t_batch_idx*path_cnt*in2_size + exec_idx*in2_size;
    in2_idx = g_dT_din2 + threadIdx.x;
    for(int i = threadIdx.x; in2_idx < g_dT_din2+in2_size; i+=WARPSIZE, in2_idx+=WARPSIZE) {
        mem_dT_din2[in2_idx] = my_shmem_dT_din2[i];
        mem_dT_dgy[in2_idx] = my_shmem_dT_dgy[i];
    }
}

void triple_bwd_sptp_linear_cuda_v2_shared_exp(
    torch::Tensor mem_dT_dGin1, 
    torch::Tensor mem_dT_dGin2,
    torch::Tensor mem_dT_dGW,
    torch::Tensor mem_dT_dGLo,

    torch::Tensor mem_dL_dO,
    torch::Tensor mem_dG_dLx,
    torch::Tensor mem_dG_dLy,
    torch::Tensor mem_dG_dLW,

    torch::Tensor in1,
    torch::Tensor in2,
    torch::Tensor weight,
    
    torch::Tensor per_edge_src,
    torch::Tensor per_edge_dst,

    torch::Tensor mem_dT_din1,
    torch::Tensor mem_dT_din2,
    torch::Tensor mem_dT_dW,
    torch::Tensor mem_dT_dLo,
    torch::Tensor mem_dT_dgx,
    torch::Tensor mem_dT_dgy,
    torch::Tensor mem_dT_dgW,
    torch::Tensor mem_debug,

    torch::Tensor t_in1_idxing,
    torch::Tensor t_in1_ival,
    torch::Tensor t_in1_related_path_idx,

    torch::Tensor t_path_array1,
    torch::Tensor t_path_array2,
    torch::Tensor t_per_path_fiber_start,
    torch::Tensor t_path_weight,
    torch::Tensor t_per_path_weight_pos,

    torch::Tensor t_fiber_array,
    torch::Tensor t_unique_cg_val,
    torch::Tensor t_per_exec_info,

    size_t path_cnt,
    size_t per_block_batch,
    size_t max_ir_dim
    ){

    // TODO: not transposed (z, mul, ir)
    const auto batch_size = in2.size(0);
    const auto in1_size = in1.size(1);
    const auto in2_size = in2.size(1);
    int perwarp_in2_size = in2_size;
    if (in2_size%2 ==0) perwarp_in2_size = in2_size+1;
    const auto out_size = mem_dT_dLo.size(1);
    const auto weight_size = weight.size(1);
    const auto batch_block = (int) std::ceil((float)batch_size/(float)per_block_batch);
    dim3 grid(batch_block, path_cnt);
    dim3 block(WARPSIZE, per_block_batch);

    // setup constant memory 
    cudaMemcpyToSymbol(in1_idxing, t_in1_idxing.data<int>(), at::numel(t_in1_idxing)*sizeof(int)); // int , MAX_IN1_IR_CNT
    cudaMemcpyToSymbol(in1_ival, t_in1_ival.data<int>(),  at::numel(t_in1_ival)*sizeof(int)); // int , MAX_IN1_IR_CNT
    cudaMemcpyToSymbol(in1_related_path_idx, t_in1_related_path_idx.data<int>(), at::numel(t_in1_related_path_idx)*sizeof(int)); // int  , MAX_IN1_IR_CNT
    
    cudaMemcpyToSymbol(path_array1, t_path_array1.data<u_int>(), at::numel(t_path_array1)*sizeof(u_int) ); // ushort2, MAX_NUM_PATH
    cudaMemcpyToSymbol(path_array2, t_path_array2.data<u_char>(), at::numel(t_path_array2)*sizeof(u_char)); // uchar4, MAX_NUM_PAT
    cudaMemcpyToSymbol(per_path_fiber_start, t_per_path_fiber_start.data<u_short>(), at::numel(t_per_path_fiber_start)*sizeof(u_short)); // ushort2, MAX_NUM_PATH
    cudaMemcpyToSymbol(path_weight, t_path_weight.data<float>(), at::numel(t_path_weight)*sizeof(float)); // float, MAX_NUM_PATH
    cudaMemcpyToSymbol(per_path_weight_pos, t_per_path_weight_pos.data<int>(), at::numel(t_per_path_weight_pos)*sizeof(int)); // int , MAX_NUM_PATH

    cudaMemcpyToSymbol(fiber_array, t_fiber_array.data<u_char>(), at::numel(t_fiber_array)*sizeof(u_char)); // u_char4), MAX_U_FIBER_CNT 
    cudaMemcpyToSymbol(unique_cg_val, t_unique_cg_val.data<float>(), at::numel(t_unique_cg_val)*sizeof(float) ); // float , MAX_U_CG_VAL_CNT

    const int shared_memory_bytes = sizeof(float) * per_block_batch * (WARPSIZE * (max_ir_dim*12+perwarp_in2_size*2) + in2_size*3);

    // int carveout = 100;
    // CHECK_CUDA_ERROR(cudaFuncSetAttribute(
    //     sptp_all_forward_kernel_v1<float>,
    //     cudaFuncAttributePreferredSharedMemoryCarveout, carveout));

    CHECK_CUDA_ERROR(cudaFuncSetAttribute(
        triple_bwd_sptp_lienar_kernel_v2_shared_exp<float>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory_bytes));

    triple_bwd_sptp_lienar_kernel_v2_shared_exp<float><<<grid, block, shared_memory_bytes>>>(
        mem_dT_dGin1.data<float>(),
        mem_dT_dGin2.data<float>(),
        mem_dT_dGW.data<float>(),
        mem_dT_dGLo.data<float>(),

        mem_dL_dO.data<float>(),
        mem_dG_dLx.data<float>(),
        mem_dG_dLy.data<float>(),
        mem_dG_dLW.data<float>(),

        in1.data<float>(),
        in2.data<float>(),
        weight.data<float>(),

        per_edge_src.data<int>(),
        per_edge_dst.data<int>(),

        mem_dT_din1.data<float>(),
        mem_dT_din2.data<float>(),
        mem_dT_dW.data<float>(),
        mem_dT_dLo.data<float>(),
        mem_dT_dgx.data<float>(),
        mem_dT_dgy.data<float>(),
        mem_dT_dgW.data<float>(),
        mem_debug.data<float>(),
        t_per_exec_info.data<u_short>(),
        
        batch_size,
        out_size,
        weight_size,
        in1_size,
        in2_size,
        perwarp_in2_size,
        path_cnt,
        max_ir_dim
        );
}
    

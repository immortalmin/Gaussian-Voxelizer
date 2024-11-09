#pragma once

#include <iostream>
#include <vector>
#include "voxelizer.h"
#include <cuda_runtime_api.h>

namespace CudaVoxelizer
{
	template <typename T>
	static void obtain(char*& chunk, T*& ptr, std::size_t count, std::size_t alignment)
	{
		std::size_t offset = (reinterpret_cast<std::uintptr_t>(chunk) + alignment - 1) & ~(alignment - 1);
		ptr = reinterpret_cast<T*>(offset);
		chunk = reinterpret_cast<char*>(ptr + count);
	}

	struct GeometryState
	{
		size_t scan_size;
		//float* depths;
		char* scanning_space;
		//bool* clamped;
		float* internal_radii;
		float* cov3Ds;
        float* cov3Ds_inv;
		//float4* conic_density;
        //float2* coefs;
		uint32_t* point_offsets;
		uint32_t* blocks_touched;

		static GeometryState fromChunk(char*& chunk, size_t P);
	};

	struct ImageState
	{
		uint2* ranges;
		//uint32_t* n_contrib;

		static ImageState fromChunk(char*& chunk, size_t N);
	};

	struct BinningState
	{
		size_t sorting_size;
		uint32_t* point_list_keys_unsorted;
		uint32_t* point_list_keys;
		uint32_t* point_list_unsorted;
		uint32_t* point_list;
		char* list_sorting_space;

		static BinningState fromChunk(char*& chunk, size_t P);
	};

	template<typename T> 
	size_t required(size_t P)
	{
		char* size = nullptr;
		T::fromChunk(size, P);
		return ((size_t)size) + 128;
	}
};

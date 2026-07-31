/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "worker/rocshmem/rocshmem_worker.h"
#include "runtime/runtime.h"
#include "utils/utils.h"
#include <iostream>
#include <cstring>

#if HAVE_ROCSHMEM && HAVE_ROCM
#define CHECK_ROCSHMEM_ERROR(result, message)                                       \
    do {                                                                            \
        if (0 != result) {                                                          \
            std::cerr << "rocSHMEM: " << message << " (Error code: " << result     \
                      << ")" << std::endl;                                          \
            exit(EXIT_FAILURE);                                                     \
        }                                                                           \
    } while(0)

#define CHECK_HIP_ERROR(result, message)                                            \
    do {                                                                            \
        const auto _r = (result);                                                   \
        if (_r != hipSuccess) {                                                     \
            std::cerr << "HIP: " << message << " (Error code: " << _r << " - "     \
                      << hipGetErrorString(_r) << ")" << std::endl;                 \
            exit(EXIT_FAILURE);                                                     \
        }                                                                           \
    } while(0)

xferBenchRocshmemWorker::xferBenchRocshmemWorker() : xferBenchWorker() {
    if (XFERBENCH_RT_ETCD == xferBenchConfig::runtime_type) {
        rank = rt->getRank();
        size = rt->getSize();

        return;        //rocSHMEM not yet initialized
    }

    std::cout << "Runtime " << xferBenchConfig::runtime_type
              << " not supported for rocSHMEM worker" << std::endl;
    exit(EXIT_FAILURE);
}

xferBenchRocshmemWorker::~xferBenchRocshmemWorker() {
    std::cout << "rocSHMEM: calling rocshmem_finalize" << std::endl;
    rocshmem_finalize();
    std::cout << "rocSHMEM: rocshmem_finalize done" << std::endl;
}

std::optional<xferBenchIOV> xferBenchRocshmemWorker::initBasicDescRocshmem(size_t buffer_size, int mem_dev_id) {
    void *addr;

    std::cout << "rocSHMEM: calling rocshmem_malloc size=" << buffer_size << std::endl;
    addr = rocshmem_malloc(buffer_size);
    std::cout << "rocSHMEM: rocshmem_malloc returned addr=" << addr << std::endl;
    if (!addr) {
        std::cerr << "Failed to allocate " << buffer_size << " bytes of rocSHMEM memory" << std::endl;
        return std::nullopt;
    }

    if (isInitiator()) {
        CHECK_HIP_ERROR(hipMemset(addr, XFERBENCH_INITIATOR_BUFFER_ELEMENT, buffer_size),
                        "Failed to memset initiator buffer");
        CHECK_HIP_ERROR(hipStreamSynchronize(0), "Failed to synchronize stream after initiator memset");
    } else if (isTarget()) {
        CHECK_HIP_ERROR(hipMemset(addr, XFERBENCH_TARGET_BUFFER_ELEMENT, buffer_size),
                        "Failed to memset target buffer");
        CHECK_HIP_ERROR(hipStreamSynchronize(0), "Failed to synchronize stream after target memset");
    }

    return std::optional<xferBenchIOV>(std::in_place, (uintptr_t)addr, buffer_size, mem_dev_id);
}

void xferBenchRocshmemWorker::cleanupBasicDescRocshmem(xferBenchIOV &iov) {
    std::cout << "rocSHMEM: calling rocshmem_free addr=" << (void *)iov.addr << std::endl;
    rocshmem_free((void *)iov.addr);
    std::cout << "rocSHMEM: rocshmem_free done" << std::endl;
}

std::vector<std::vector<xferBenchIOV>> xferBenchRocshmemWorker::allocateMemory(int num_threads) {
    std::vector<std::vector<xferBenchIOV>> iov_lists;
    size_t i, buffer_size, num_devices = 0;

    if (1 != num_threads) {
        std::cerr << "rocSHMEM: Only 1 thread is supported for now" << std::endl;
        exit(EXIT_FAILURE);
    }

    if (isInitiator()) {
        num_devices = xferBenchConfig::num_initiator_dev;
    } else if (isTarget()) {
        num_devices = xferBenchConfig::num_target_dev;
    }
    buffer_size = xferBenchConfig::total_buffer_size / (num_devices * num_threads);

    for (int list_idx = 0; list_idx < num_threads; list_idx++) {
        std::vector<xferBenchIOV> iov_list;
        for (i = 0; i < num_devices; i++) {
            std::optional<xferBenchIOV> basic_desc;
            basic_desc = initBasicDescRocshmem(buffer_size, i);
            if (basic_desc) {
                iov_list.push_back(basic_desc.value());
            }
        }
        iov_lists.push_back(iov_list);
    }
    return iov_lists;
}

void xferBenchRocshmemWorker::deallocateMemory(std::vector<std::vector<xferBenchIOV>> &iov_lists) {
    std::cout << "rocSHMEM: calling rocshmem_barrier_all (deallocate)" << std::endl;
    rocshmem_barrier_all();
    std::cout << "rocSHMEM: rocshmem_barrier_all done (deallocate)" << std::endl;
    for (auto &iov_list: iov_lists) {
        for (auto &iov: iov_list) {
            cleanupBasicDescRocshmem(iov);
        }
    }
}

int xferBenchRocshmemWorker::exchangeMetadata() {
    // No metadata exchange needed for rocSHMEM
    return 0;
}

std::vector<std::vector<xferBenchIOV>>
xferBenchRocshmemWorker::exchangeIOV(const std::vector<std::vector<xferBenchIOV>> &iov_lists,
                                     size_t block_size) {
    // For rocSHMEM, we don't need to exchange IOV lists
    // This will just return local IOV list
    return iov_lists;
}

// No thread support for rocSHMEM yet
static int
execTransfer(const std::vector<std::vector<xferBenchIOV>> &local_iovs,
             const std::vector<std::vector<xferBenchIOV>> &remote_iovs,
             const int num_iter,
             hipStream_t stream,
             xferBenchStats &stats) {
    int ret = 0, tid = 0, target_rank;

    target_rank = 1;

    const auto &local_iov = local_iovs[tid];
    const auto &remote_iov = remote_iovs[tid];

    xferBenchTimer total_timer;
    xferBenchTimer timer;

    for (int i = 0; i < num_iter; i++) {
        for (size_t i = 0; i < local_iov.size(); i++) {
            auto &local = local_iov[i];
            auto &remote = remote_iov[i];
            if (XFERBENCH_OP_WRITE == xferBenchConfig::op_type) {
                std::cout << "rocSHMEM: calling rocshmem_putmem_on_stream len=" << local.len
                          << " pe=" << target_rank << std::endl;
                rocshmem_putmem_on_stream(
                    (void *)remote.addr, (void *)local.addr, local.len, target_rank, stream);
                std::cout << "rocSHMEM: rocshmem_putmem_on_stream done" << std::endl;
            } else if (XFERBENCH_OP_READ == xferBenchConfig::op_type) {
                std::cout << "rocSHMEM: calling rocshmem_getmem_on_stream len=" << local.len
                          << " pe=" << target_rank << std::endl;
                rocshmem_getmem_on_stream(
                    (void *)remote.addr, (void *)local.addr, local.len, target_rank, stream);
                std::cout << "rocSHMEM: rocshmem_getmem_on_stream done" << std::endl;
            }
        }
        std::cout << "rocSHMEM: calling rocshmem_quiet_on_stream" << std::endl;
        rocshmem_quiet_on_stream(stream);
        std::cout << "rocSHMEM: rocshmem_quiet_on_stream done" << std::endl;
        nixlTime::us_t transfer_duration = timer.lap();
        stats.transfer_duration.add(transfer_duration);
    }

    nixlTime::us_t total_duration = total_timer.lap();
    stats.total_duration.add(total_duration);

    return ret;
}

std::variant<xferBenchStats, int>
xferBenchRocshmemWorker::transfer(size_t block_size,
                                  const std::vector<std::vector<xferBenchIOV>> &local_trans_lists,
                                  const std::vector<std::vector<xferBenchIOV>> &remote_trans_lists) {
    hipEvent_t start_event, stop_event;
    int num_iter = xferBenchConfig::num_iter / xferBenchConfig::num_threads;
    int skip = xferBenchConfig::warmup_iter / xferBenchConfig::num_threads;
    int ret = 0;
    xferBenchStats stats;

    // Create events to time the transfer
    CHECK_HIP_ERROR(hipEventCreate(&start_event), "Failed to create HIP event");
    CHECK_HIP_ERROR(hipEventCreate(&stop_event), "Failed to create HIP event");

    // Here the local_trans_lists is the same as remote_trans_lists
    // Reduce skip by 10x for large block sizes
    if (block_size > LARGE_BLOCK_SIZE) {
        skip /= xferBenchConfig::large_blk_iter_ftr;
        num_iter /= xferBenchConfig::large_blk_iter_ftr;
    }

    if (skip > 0) {
        ret = execTransfer(local_trans_lists, remote_trans_lists, skip, stream, stats);
        if (ret < 0) {
            return std::variant<xferBenchStats, int>(ret);
        }
        stats.clear();
    }
    std::cout << "rocSHMEM: calling rocshmem_barrier_all_on_stream (pre-transfer)" << std::endl;
    rocshmem_barrier_all_on_stream(stream);
    std::cout << "rocSHMEM: rocshmem_barrier_all_on_stream done (pre-transfer)" << std::endl;
    CHECK_HIP_ERROR(hipStreamSynchronize(stream), "Failed to synchronize HIP stream");

    CHECK_HIP_ERROR(hipEventRecord(start_event, stream), "Failed to record HIP event");

    ret = execTransfer(local_trans_lists, remote_trans_lists, num_iter, stream, stats);

    CHECK_HIP_ERROR(hipEventRecord(stop_event, stream), "Failed to record HIP event");

    std::cout << "rocSHMEM: calling rocshmem_barrier_all_on_stream (post-transfer)" << std::endl;
    rocshmem_barrier_all_on_stream(stream);
    std::cout << "rocSHMEM: rocshmem_barrier_all_on_stream done (post-transfer)" << std::endl;
    CHECK_HIP_ERROR(hipEventSynchronize(stop_event), "Failed to synchronize HIP event");
    CHECK_HIP_ERROR(hipStreamSynchronize(stream), "Failed to synchronize HIP stream");

    return ret < 0 ? std::variant<xferBenchStats, int>(ret) :
                     std::variant<xferBenchStats, int>(stats);
}

void
xferBenchRocshmemWorker::poll(size_t block_size) {
    // For rocSHMEM, we don't need to poll
    // The transfer is already complete when we reach this point
    std::cout << "rocSHMEM: calling rocshmem_barrier_all_on_stream (poll 1)" << std::endl;
    rocshmem_barrier_all_on_stream(stream);
    std::cout << "rocSHMEM: rocshmem_barrier_all_on_stream done (poll 1)" << std::endl;
    CHECK_HIP_ERROR(hipStreamSynchronize(stream), "Failed to synchronize HIP stream");

    std::cout << "rocSHMEM: calling rocshmem_barrier_all_on_stream (poll 2)" << std::endl;
    rocshmem_barrier_all_on_stream(stream);
    std::cout << "rocSHMEM: rocshmem_barrier_all_on_stream done (poll 2)" << std::endl;
    CHECK_HIP_ERROR(hipStreamSynchronize(stream), "Failed to synchronize HIP stream");
}

int xferBenchRocshmemWorker::synchronizeStart() {
    rocshmem_init_attr_t attr = {};
    group_id = {};

    if (xferBenchConfig::runtime_type == XFERBENCH_RT_ETCD) {
        if (rank == 0 && group_id_initialized == 0) {
            std::cout << "rocSHMEM: calling rocshmem_get_uniqueid (rank 0)" << std::endl;
            rocshmem_get_uniqueid(&group_id);
            std::cout << "rocSHMEM: rocshmem_get_uniqueid done" << std::endl;
        }

        rt->broadcastInt((int *)&group_id, sizeof(rocshmem_uniqueid_t) / sizeof(int), 0);
        group_id_initialized = 1;

        std::cout << "rocSHMEM: calling rocshmem_set_attr_uniqueid_args rank=" << rank << " size=" << size << std::endl;
        rocshmem_set_attr_uniqueid_args(rank, size, &group_id, &attr);
        std::cout << "rocSHMEM: calling rocshmem_init_attr" << std::endl;
        rocshmem_init_attr(ROCSHMEM_INIT_WITH_UNIQUEID, &attr);
        std::cout << "rocSHMEM: rocshmem_init_attr done" << std::endl;

        // Create a stream
        CHECK_HIP_ERROR(hipSetDevice(rank), "Failed to set HIP device");
        CHECK_HIP_ERROR(hipStreamCreate(&stream), "Failed to create HIP stream");
        std::cout << "rocSHMEM: HIP stream created" << std::endl;
    }

    std::cout << "rocSHMEM: calling rocshmem_barrier_all_on_stream (synchronizeStart)" << std::endl;
    rocshmem_barrier_all_on_stream(stream);
    std::cout << "rocSHMEM: rocshmem_barrier_all_on_stream done (synchronizeStart)" << std::endl;

    return 0;
}

#endif

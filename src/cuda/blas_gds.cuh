#pragma once

#include <storage_gds.cuh>
#include <utils.cuh>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

void operator_add(const GDSStorage *input1, const GDSStorage *input2,
                  GDSStorage *outputs);
void operator_add(const GDSStorage *input1, float value, GDSStorage *outputs);

void operator_sub(const GDSStorage *input1, const GDSStorage *input2,
                  GDSStorage *outputs);

void operator_mul(const GDSStorage *input1, const GDSStorage *input2,
                  GDSStorage *outputs);
void operator_mul(const GDSStorage *input1, float value, GDSStorage *outputs);

void operator_div(const GDSStorage *input1, const GDSStorage *input2,
                  GDSStorage *outputs);

void operator_log(const GDSStorage *input1, GDSStorage *outputs);

void operator_exp(const GDSStorage *input1, GDSStorage *outputs);

void operator_pow(const GDSStorage *input1, float e, GDSStorage *outputs);

void operator_matmul(const GDSStorage *input1, const GDSStorage *input2,
                     GDSStorage *outputs, int broadcast = 0);

void operator_transpose(const GDSStorage *input1, GDSStorage *outputs);

void operator_mean(const GDSStorage *input1, int dim, GDSStorage *outputs);

void operator_sum(const GDSStorage *input1, int dim, GDSStorage *outputs);
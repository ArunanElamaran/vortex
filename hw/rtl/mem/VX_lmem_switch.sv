// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

`include "VX_define.vh"

module VX_lmem_switch import VX_gpu_pkg::*; #(
    parameter GLOBAL_OUT_BUF = 0,
    parameter LOCAL_OUT_BUF = 0,
    parameter RSP_OUT_BUF = 0,
    parameter TENSOR_OUT_BUF = 0,
    parameter `STRING ARBITER = "R"
) (
    input wire              clk,
    input wire              reset,
    VX_lsu_mem_if.slave     lsu_in_if,
    VX_lsu_mem_if.master    global_out_if,
    VX_lsu_mem_if.master    local_out_if
`ifdef TMEM_ENABLE
   ,VX_lsu_mem_if.master    tensor_out_if
`endif
);
    localparam REQ_DATAW = `NUM_LSU_LANES + 1 + `NUM_LSU_LANES * (LSU_WORD_SIZE + LSU_ADDR_WIDTH + MEM_FLAGS_WIDTH + LSU_WORD_SIZE * 8) + LSU_TAG_WIDTH;
    localparam RSP_DATAW = `NUM_LSU_LANES + `NUM_LSU_LANES * (LSU_WORD_SIZE * 8) + LSU_TAG_WIDTH;

    wire [`NUM_LSU_LANES-1:0] is_addr_local_mask;
`ifdef TMEM_ENABLE
    wire [`NUM_LSU_LANES-1:0] is_addr_tensor_mask;
`endif
    wire req_global_ready;
    wire req_local_ready;
`ifdef TMEM_ENABLE
    wire req_tensor_ready;
`endif

`ifdef LMEM_ENABLE
    for (genvar i = 0; i < `NUM_LSU_LANES; ++i) begin : g_is_addr_local_mask
        assign is_addr_local_mask[i] = lsu_in_if.req_data.flags[i][MEM_REQ_FLAG_LOCAL];
    end
`else
    assign is_addr_local_mask = '0;
`endif

`ifdef TMEM_ENABLE
    for (genvar i = 0; i < `NUM_LSU_LANES; ++i) begin : g_is_addr_tensor_mask
        assign is_addr_tensor_mask[i] = lsu_in_if.req_data.flags[i][MEM_REQ_FLAG_TMEM];
    end
`endif

`ifdef TMEM_ENABLE
    wire is_addr_tensor = | (lsu_in_if.req_data.mask & is_addr_tensor_mask);
`else
    wire is_addr_tensor = 1'b0;
`endif
    wire is_addr_local  = | (lsu_in_if.req_data.mask & is_addr_local_mask);
    wire is_addr_global = | (lsu_in_if.req_data.mask & ~(is_addr_local_mask
    `ifdef TMEM_ENABLE
                                                       | is_addr_tensor_mask
    `endif
                                                       ));

`ifdef TMEM_ENABLE
    `RUNTIME_ASSERT(~(is_addr_tensor && is_addr_local), ("%t: mixed TMEM/LMEM request", $time))
    `RUNTIME_ASSERT(~(is_addr_tensor && is_addr_global), ("%t: mixed TMEM/global request", $time))
`endif

    assign lsu_in_if.req_ready = (req_global_ready && is_addr_global)
                              || (req_local_ready && is_addr_local)
`ifdef TMEM_ENABLE
                              || (req_tensor_ready && is_addr_tensor)
`endif
                              ;

    VX_elastic_buffer #(
        .DATAW   (REQ_DATAW),
        .SIZE    (`TO_OUT_BUF_SIZE(GLOBAL_OUT_BUF)),
        .OUT_REG (`TO_OUT_BUF_REG(GLOBAL_OUT_BUF))
    ) req_global_buf (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (lsu_in_if.req_valid && is_addr_global),
        .data_in   ({
            lsu_in_if.req_data.mask & ~is_addr_local_mask,
            lsu_in_if.req_data.rw,
            lsu_in_if.req_data.addr,
            lsu_in_if.req_data.data,
            lsu_in_if.req_data.byteen,
            lsu_in_if.req_data.flags,
            lsu_in_if.req_data.tag
        }),
        .ready_in  (req_global_ready),
        .valid_out (global_out_if.req_valid),
        .data_out  (global_out_if.req_data),
        .ready_out (global_out_if.req_ready)
    );

    VX_elastic_buffer #(
        .DATAW   (REQ_DATAW),
        .SIZE    (`TO_OUT_BUF_SIZE(LOCAL_OUT_BUF)),
        .OUT_REG (`TO_OUT_BUF_REG(LOCAL_OUT_BUF))
    ) req_local_buf (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (lsu_in_if.req_valid && is_addr_local),
        .data_in   ({
            lsu_in_if.req_data.mask & is_addr_local_mask,
            lsu_in_if.req_data.rw,
            lsu_in_if.req_data.addr,
            lsu_in_if.req_data.data,
            lsu_in_if.req_data.byteen,
            lsu_in_if.req_data.flags,
            lsu_in_if.req_data.tag
        }),
        .ready_in  (req_local_ready),
        .valid_out (local_out_if.req_valid),
        .data_out  (local_out_if.req_data),
        .ready_out (local_out_if.req_ready)
    );

`ifdef TMEM_ENABLE

    VX_elastic_buffer #(
        .DATAW   (REQ_DATAW),
        .SIZE    (`TO_OUT_BUF_SIZE(TENSOR_OUT_BUF)),
        .OUT_REG (`TO_OUT_BUF_REG(TENSOR_OUT_BUF))
    ) req_tensor_buf (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (lsu_in_if.req_valid && is_addr_tensor),
        .data_in   ({
            lsu_in_if.req_data.mask & is_addr_tensor_mask,
            lsu_in_if.req_data.rw,
            lsu_in_if.req_data.addr,
            lsu_in_if.req_data.data,
            lsu_in_if.req_data.byteen,
            lsu_in_if.req_data.flags,
            lsu_in_if.req_data.tag
        }),
        .ready_in  (req_tensor_ready),
        .valid_out (tensor_out_if.req_valid),
        .data_out  (tensor_out_if.req_data),
        .ready_out (tensor_out_if.req_ready)
    );

    VX_stream_arb #(
        .NUM_INPUTS (3),
        .DATAW      (RSP_DATAW),
        .ARBITER    (ARBITER),
        .OUT_BUF    (RSP_OUT_BUF)
    ) rsp_arb (
        .clk       (clk),
        .reset     (reset),
        .valid_in  ({
            tensor_out_if.rsp_valid,
            local_out_if.rsp_valid,
            global_out_if.rsp_valid
        }),
        .ready_in  ({
            tensor_out_if.rsp_ready,
            local_out_if.rsp_ready,
            global_out_if.rsp_ready
        }),
        .data_in   ({
            tensor_out_if.rsp_data,
            local_out_if.rsp_data,
            global_out_if.rsp_data
        }),
        .data_out  (lsu_in_if.rsp_data),
        .valid_out (lsu_in_if.rsp_valid),
        .ready_out (lsu_in_if.rsp_ready),
        `UNUSED_PIN (sel_out)
    );

`else

    VX_stream_arb #(
        .NUM_INPUTS (2),
        .DATAW      (RSP_DATAW),
        .ARBITER    (ARBITER),
        .OUT_BUF    (RSP_OUT_BUF)
    ) rsp_arb (
        .clk       (clk),
        .reset     (reset),
        .valid_in  ({
            local_out_if.rsp_valid,
            global_out_if.rsp_valid
        }),
        .ready_in  ({
            local_out_if.rsp_ready,
            global_out_if.rsp_ready
        }),
        .data_in   ({
            local_out_if.rsp_data,
            global_out_if.rsp_data
        }),
        .data_out  (lsu_in_if.rsp_data),
        .valid_out (lsu_in_if.rsp_valid),
        .ready_out (lsu_in_if.rsp_ready),
        `UNUSED_PIN (sel_out)
    );

`endif

endmodule

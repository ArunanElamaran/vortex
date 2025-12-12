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

module VX_tcu_fp import VX_gpu_pkg::*, VX_tcu_pkg::*; #(
    parameter `STRING INSTANCE_ID = ""
) (
    `SCOPE_IO_DECL

    input wire          clk,
    input wire          reset,

    // Inputs
    VX_execute_if.slave execute_if,

    // Outputs
    VX_result_if.master result_if,

    // Tensor memory interface (single port)
    VX_mem_bus_if.master tcu_tmem_bus_if
);

    // -------------------------------------------------------------------------
    // UMMA with TMEM backing (single in-flight op)
    // -------------------------------------------------------------------------

    localparam WORD_BYTES = LSU_WORD_SIZE;
    localparam WORD_BITS  = WORD_BYTES * 8;
    localparam TMEM_ADDRW = `MEM_ADDR_WIDTH - `CLOG2(LSU_WORD_SIZE);

    typedef enum logic [2:0] {
        S_IDLE,
        S_READ_A,
        S_READ_B,
        S_READ_C,
        S_COMPUTE,
        S_WRITE_D,
        S_RESP
    } umma_state_t;

    umma_state_t state;

    // Latched instruction metadata
    reg [3:0] umma_fmt_s, umma_fmt_d;
    reg [TMEM_ADDRW-1:0] addr_b, addr_c;
    reg [UUID_WIDTH-1:0] umma_uuid;
    reg [NW_WIDTH-1:0]    umma_wid;
    reg [PC_BITS-1:0]     umma_pc;

    // Tile sizing
    wire [31:0] tileM = TCU_TILE_M;
    wire [31:0] tileN = TCU_TILE_N;
    wire [31:0] tileK_base = TCU_TILE_K;
    wire [31:0] itype_bytes = tcu_type_bytes(umma_fmt_s);
    wire [31:0] ratio = (4 / itype_bytes);
    wire [31:0] tileK = tileK_base * ratio;
    wire [31:0] otype_bytes = tcu_type_bytes(umma_fmt_d);

    // Buffer sizes in bytes
    wire [31:0] a_bytes = tileM * tileK * itype_bytes;
    wire [31:0] b_bytes = tileK * tileN * itype_bytes;
    wire [31:0] c_bytes = tileM * tileN * otype_bytes;

    // Max buffers (for current configs, tileM<=8, tileN<=4, tileK<=16)
    localparam MAX_A_BYTES = 8 * 16 * 4;
    localparam MAX_B_BYTES = 16 * 4 * 4;
    localparam MAX_C_BYTES = 8 * 4 * 4;

    reg [7:0] a_buf [0:MAX_A_BYTES-1];
    reg [7:0] b_buf [0:MAX_B_BYTES-1];
    reg [7:0] c_buf [0:MAX_C_BYTES-1];
    reg [7:0] d_buf [0:MAX_C_BYTES-1];

    // TMEM word transfer helpers
    reg req_pending;
    reg req_rw;
    reg [TMEM_ADDRW-1:0] req_addr;
    reg [WORD_BITS-1:0] req_wdata;
    reg [WORD_BYTES-1:0] req_byteen;

    wire rsp_fire = tcu_tmem_bus_if.rsp_valid && tcu_tmem_bus_if.rsp_ready;
    wire req_fire = tcu_tmem_bus_if.req_valid && tcu_tmem_bus_if.req_ready;

    assign tcu_tmem_bus_if.req_valid = req_pending;
    assign tcu_tmem_bus_if.req_data.rw = req_rw;
    assign tcu_tmem_bus_if.req_data.addr = req_addr;
    assign tcu_tmem_bus_if.req_data.data = req_wdata;
    assign tcu_tmem_bus_if.req_data.byteen = req_byteen;
    assign tcu_tmem_bus_if.req_data.flags = '0;
    assign tcu_tmem_bus_if.req_data.tag.value = '0;
    assign tcu_tmem_bus_if.req_data.tag.uuid  = '0;
    assign tcu_tmem_bus_if.rsp_ready = 1'b1;

    // Word index counters
    reg [31:0] word_idx;
    reg [31:0] total_words;
    reg [TMEM_ADDRW-1:0] base_addr_latched;

    // Ready/valid gating
    wire umma_req = execute_if.valid
                 && (execute_if.data.op_type == INST_TCU_UMMA);
    wire accept_op = (state == S_IDLE) && umma_req;
    wire umma_active = (state != S_IDLE);

    // UMMA result interface (only active when state == S_RESP)
    wire umma_result_valid = (state == S_RESP);

    // Byte pack/unpack helpers (buffer-specific to avoid SV array formals issues)
    task automatic store_a(input integer off, input [WORD_BITS-1:0] wdata, input integer nbytes);
        integer bi;
        begin
            for (bi = 0; bi < nbytes; bi = bi + 1)
                a_buf[off + bi] <= wdata[8*bi +: 8];
        end
    endtask
    task automatic store_b(input integer off, input [WORD_BITS-1:0] wdata, input integer nbytes);
        integer bi;
        begin
            for (bi = 0; bi < nbytes; bi = bi + 1)
                b_buf[off + bi] <= wdata[8*bi +: 8];
        end
    endtask
    task automatic store_c(input integer off, input [WORD_BITS-1:0] wdata, input integer nbytes);
        integer bi;
        begin
            for (bi = 0; bi < nbytes; bi = bi + 1)
                c_buf[off + bi] <= wdata[8*bi +: 8];
        end
    endtask
    task automatic store_d(input integer off, input [WORD_BITS-1:0] wdata, input integer nbytes);
        integer bi;
        begin
            for (bi = 0; bi < nbytes; bi = bi + 1)
                d_buf[off + bi] <= wdata[8*bi +: 8];
        end
    endtask

    function automatic [WORD_BITS-1:0] load_d(input integer off, input integer nbytes);
        integer bi;
        reg [WORD_BITS-1:0] tmp;
        begin
            tmp = '0;
            for (bi = 0; bi < nbytes; bi = bi + 1)
                tmp[8*bi +: 8] = d_buf[off + bi];
            load_d = tmp;
        end
    endfunction

    // Float helpers
    function automatic [31:0] load_a_word(input integer off);
        load_a_word = {a_buf[off+3], a_buf[off+2], a_buf[off+1], a_buf[off+0]};
    endfunction
    function automatic [31:0] load_b_word(input integer off);
        load_b_word = {b_buf[off+3], b_buf[off+2], b_buf[off+1], b_buf[off+0]};
    endfunction
    function automatic [31:0] load_c_word(input integer off);
        load_c_word = {c_buf[off+3], c_buf[off+2], c_buf[off+1], c_buf[off+0]};
    endfunction

    function automatic real word_to_real(input [31:0] w);
        word_to_real = $bitstoreal({32'b0, w}); // use real; upper bits zero
    endfunction
    function automatic [31:0] real_to_word(input real v);
        real_to_word = $realtobits(v)[31:0];
    endfunction

    // Compute D = C + A*B
    task automatic do_compute;
        integer m, n, k;
        integer a_off, b_off, c_off, d_off;
        real acc;
        real aval, bval;
        begin
            for (m = 0; m < tileM; m = m + 1) begin
                for (n = 0; n < tileN; n = n + 1) begin
                    c_off = (m*tileN + n)*otype_bytes;
                    acc = word_to_real(load_c_word(c_off));
                    for (k = 0; k < tileK; k = k + 1) begin
                        a_off = (m*tileK + k)*itype_bytes;
                        b_off = (k*tileN + n)*itype_bytes;
                        aval = word_to_real(load_a_word(a_off));
                        bval = word_to_real(load_b_word(b_off));
                        acc = acc + (aval * bval);
                    end
                    d_off = (m*tileN + n)*otype_bytes;
                    store_d(d_off, real_to_word(acc), otype_bytes);
                end
            end
        end
    endtask

    // FSM
    always @(posedge clk) begin
        if (reset) begin
            state <= S_IDLE;
            req_pending <= 0;
            req_rw <= 0;
            req_addr <= '0;
            req_wdata <= '0;
            req_byteen <= '0;
            word_idx <= 0;
            total_words <= 0;
            base_addr_latched <= 0;
        end else begin
            // clear issued requests when fired
            if (req_fire)
                req_pending <= 0;

            case (state)
            S_IDLE: begin
                if (accept_op) begin
                    umma_fmt_s <= execute_if.data.op_args.tcu.fmt_s;
                    umma_fmt_d <= execute_if.data.op_args.tcu.fmt_d;
                    addr_b <= execute_if.data.rs2_data[0][TMEM_ADDRW-1:0];
                    addr_c <= execute_if.data.rs3_data[0][TMEM_ADDRW-1:0];
                    umma_uuid <= execute_if.data.uuid;
                    umma_wid  <= execute_if.data.wid;
                    umma_pc   <= execute_if.data.PC;
                    // prep A read
                    base_addr_latched <= execute_if.data.rs1_data[0][TMEM_ADDRW-1:0];
                    word_idx <= 0;
                    total_words <= (a_bytes + WORD_BYTES - 1) / WORD_BYTES;
                    state <= S_READ_A;
                end
            end

            S_READ_A: begin
                if (~req_pending && (word_idx < total_words)) begin
                    integer off = word_idx * WORD_BYTES;
                    integer rem = a_bytes - off;
                    integer nbytes = (rem >= WORD_BYTES) ? WORD_BYTES : rem;
                    req_rw <= 0;
                    req_addr <= base_addr_latched[TMEM_ADDRW-1:0] + TMEM_ADDRW'(word_idx);
                    req_byteen <= (1 << nbytes) - 1;
                    req_pending <= 1;
                end
                if (rsp_fire) begin
                    integer off = word_idx * WORD_BYTES;
                    integer rem = a_bytes - off;
                    integer nbytes = (rem >= WORD_BYTES) ? WORD_BYTES : rem;
                    store_a(off, tcu_tmem_bus_if.rsp_data.data, nbytes);
                    word_idx <= word_idx + 1;
                    if (word_idx + 1 == total_words) begin
                        // move to B
                        base_addr_latched <= TMEM_ADDRW'(addr_b);
                        word_idx <= 0;
                        total_words <= (b_bytes + WORD_BYTES - 1) / WORD_BYTES;
                        state <= S_READ_B;
                    end
                end
            end

            S_READ_B: begin
                if (~req_pending && (word_idx < total_words)) begin
                    integer off = word_idx * WORD_BYTES;
                    integer rem = b_bytes - off;
                    integer nbytes = (rem >= WORD_BYTES) ? WORD_BYTES : rem;
                    req_rw <= 0;
                    req_addr <= base_addr_latched[TMEM_ADDRW-1:0] + TMEM_ADDRW'(word_idx);
                    req_byteen <= (1 << nbytes) - 1;
                    req_pending <= 1;
                end
                if (rsp_fire) begin
                    integer off = word_idx * WORD_BYTES;
                    integer rem = b_bytes - off;
                    integer nbytes = (rem >= WORD_BYTES) ? WORD_BYTES : rem;
                    store_b(off, tcu_tmem_bus_if.rsp_data.data, nbytes);
                    word_idx <= word_idx + 1;
                    if (word_idx + 1 == total_words) begin
                        // move to C
                        base_addr_latched <= TMEM_ADDRW'(addr_c);
                        word_idx <= 0;
                        total_words <= (c_bytes + WORD_BYTES - 1) / WORD_BYTES;
                        state <= S_READ_C;
                    end
                end
            end

            S_READ_C: begin
                if (~req_pending && (word_idx < total_words)) begin
                    integer off = word_idx * WORD_BYTES;
                    integer rem = c_bytes - off;
                    integer nbytes = (rem >= WORD_BYTES) ? WORD_BYTES : rem;
                    req_rw <= 0;
                    req_addr <= base_addr_latched[TMEM_ADDRW-1:0] + TMEM_ADDRW'(word_idx);
                    req_byteen <= (1 << nbytes) - 1;
                    req_pending <= 1;
                end
                if (rsp_fire) begin
                    integer off = word_idx * WORD_BYTES;
                    integer rem = c_bytes - off;
                    integer nbytes = (rem >= WORD_BYTES) ? WORD_BYTES : rem;
                    store_c(off, tcu_tmem_bus_if.rsp_data.data, nbytes);
                    word_idx <= word_idx + 1;
                    if (word_idx + 1 == total_words) begin
                        state <= S_COMPUTE;
                    end
                end
            end

            S_COMPUTE: begin
                do_compute();
                // prep D write
                base_addr_latched <= TMEM_ADDRW'(addr_c);
                word_idx <= 0;
                total_words <= (c_bytes + WORD_BYTES - 1) / WORD_BYTES;
                state <= S_WRITE_D;
            end

            S_WRITE_D: begin
                if (~req_pending && (word_idx < total_words)) begin
                    integer off = word_idx * WORD_BYTES;
                    integer rem = c_bytes - off;
                    integer nbytes = (rem >= WORD_BYTES) ? WORD_BYTES : rem;
                    req_rw <= 1;
                    req_addr <= base_addr_latched[TMEM_ADDRW-1:0] + TMEM_ADDRW'(word_idx);
                    req_byteen <= (1 << nbytes) - 1;
                    req_wdata <= load_d(off, nbytes);
                    req_pending <= 1;
                end
                if (req_fire) begin
                    word_idx <= word_idx + 1;
                    if (word_idx + 1 == total_words) begin
                        state <= S_RESP;
                    end
                end
            end

            S_RESP: begin
                if (result_if.ready) begin
                    state <= S_IDLE;
                end
            end

            default: state <= S_IDLE;
            endcase
        end
    end
    `UNUSED_SPARAM (INSTANCE_ID);

    localparam MDATA_WIDTH = UUID_WIDTH + NW_WIDTH + PC_BITS + NUM_REGS_BITS;

`ifdef TCU_DSP
    localparam FCVT_LATENCY = 1;
    localparam FMUL_LATENCY = 8;
    localparam FADD_LATENCY = 11;
    localparam FACC_LATENCY = $clog2(2 * TCU_TC_K + 1) * FADD_LATENCY;
    localparam FEDP_LATENCY = FCVT_LATENCY + FMUL_LATENCY + FACC_LATENCY;
`elsif TCU_DPI
    localparam FMUL_LATENCY = 2;
    localparam FACC_LATENCY = 2;
    localparam FEDP_LATENCY = FMUL_LATENCY + FACC_LATENCY;
`elsif TCU_BHF
    localparam FMUL_LATENCY = 2;
    localparam FADD_LATENCY = 2;
    localparam FRND_LATENCY = 1;
    localparam FACC_LATENCY  = $clog2(2 * TCU_TC_K + 1) * (FADD_LATENCY + FRND_LATENCY);
    localparam FEDP_LATENCY = (FMUL_LATENCY + FRND_LATENCY) + 1 + FACC_LATENCY;
`endif

    localparam PIPE_LATENCY = FEDP_LATENCY + 1;
    localparam MDATA_QUEUE_DEPTH = 1 << $clog2(PIPE_LATENCY);

    localparam LG_A_BS = $clog2(TCU_A_BLOCK_SIZE);
    localparam LG_B_BS = $clog2(TCU_B_BLOCK_SIZE);
    localparam OFF_W   = $clog2(TCU_BLOCK_CAP);

    wire [3:0] step_m = execute_if.data.op_args.tcu.step_m;
    wire [3:0] step_n = execute_if.data.op_args.tcu.step_n;

    wire [3:0] fmt_s = execute_if.data.op_args.tcu.fmt_s;
    wire [3:0] fmt_d = execute_if.data.op_args.tcu.fmt_d;

    `UNUSED_VAR ({step_m, step_n, fmt_s, fmt_d});

    wire [MDATA_WIDTH-1:0] mdata_queue_din, mdata_queue_dout;
    wire mdata_queue_full;

    assign mdata_queue_din = {
        execute_if.data.uuid,
        execute_if.data.wid,
        execute_if.data.PC,
        execute_if.data.rd
    };

    wire execute_fire = execute_if.valid && execute_if.ready;
    wire fedp_enable, fedp_done;
    
    // Only use mdata_queue and FEDP for WMMA operations (UMMA uses latched metadata and direct TMEM access)
    wire wmma_execute_fire = execute_fire && (execute_if.data.op_type == INST_TCU_WMMA);
    wire wmma_result_fire = wmma_result_valid && result_if.ready;

    // FEDP delay handling
    reg [PIPE_LATENCY-1:0] fedp_delay_pipe;
    always @(posedge clk) begin
        if (reset) begin
            fedp_delay_pipe <= '0;
        end else begin
            if (fedp_enable) begin
                fedp_delay_pipe <= fedp_delay_pipe >> 1;
            end
            if (wmma_execute_fire) begin
                fedp_delay_pipe[PIPE_LATENCY-1] <= 1;
            end
        end
    end
    assign fedp_done = fedp_delay_pipe[0];

    // WMMA result valid (only when UMMA is not active)
    wire wmma_result_valid = fedp_done && (state == S_IDLE);

    assign fedp_enable      = (~umma_active) && (~wmma_result_valid || result_if.ready);
    // execute_if.ready: When UMMA active (state != S_IDLE), block new ops; when idle, allow WMMA if no UMMA req, or accept UMMA if req
    wire wmma_ready = ~mdata_queue_full && fedp_enable;
    assign execute_if.ready = (state == S_IDLE) ? (~umma_req ? wmma_ready : accept_op) : 1'b0;

    VX_fifo_queue #(
        .DATAW (MDATA_WIDTH),
        .DEPTH (MDATA_QUEUE_DEPTH),
        .OUT_REG (1)
    ) mdata_queue (
        .clk    (clk),
        .reset  (reset),
        .push   (wmma_execute_fire),
        .pop    (wmma_result_fire),
        .data_in(mdata_queue_din),
        .data_out(mdata_queue_dout),
        `UNUSED_PIN(empty),
        `UNUSED_PIN(alm_empty),
        .full   (mdata_queue_full),
        `UNUSED_PIN(alm_full),
        `UNUSED_PIN(size)
    );

    wire [OFF_W-1:0] a_off = (OFF_W'(step_m) & OFF_W'(TCU_A_SUB_BLOCKS-1)) << LG_A_BS;
    wire [OFF_W-1:0] b_off = (OFF_W'(step_n) & OFF_W'(TCU_B_SUB_BLOCKS-1)) << LG_B_BS;

    wire [TCU_TC_M-1:0][TCU_TC_N-1:0][`XLEN-1:0] d_val;

    for (genvar i = 0; i < TCU_TC_M; ++i) begin : g_i
        for (genvar j = 0; j < TCU_TC_N; ++j) begin : g_j

            wire [TCU_TC_K-1:0][`XLEN-1:0] a_row = execute_if.data.rs1_data[a_off + i * TCU_TC_K +: TCU_TC_K];
            wire [TCU_TC_K-1:0][`XLEN-1:0] b_col = execute_if.data.rs2_data[b_off + j * TCU_TC_K +: TCU_TC_K];
            wire [`XLEN-1:0] c_val = execute_if.data.rs3_data[i * TCU_TC_N + j];

            wire [2:0] fmt_s_r, fmt_d_r;
            wire [TCU_TC_K-1:0][`XLEN-1:0] a_row_r, b_col_r;
            wire [`XLEN-1:0] c_val_r;

            `BUFFER_EX (
                {a_row_r, b_col_r, c_val_r, fmt_s_r,    fmt_d_r},
                {a_row,   b_col,   c_val,   fmt_s[2:0], fmt_d[2:0]},
                fedp_enable,
                0, // resetw
                1  // depth
            );

        `ifdef TCU_DPI
            VX_tcu_fedp_dpi #(
                .LATENCY (FEDP_LATENCY),
                .N (TCU_TC_K)
            ) fedp (
                .clk   (clk),
                .reset (reset),
                .enable(fedp_enable),
                .fmt_s (fmt_s_r),
                .fmt_d (fmt_d_r),
                .a_row (a_row_r),
                .b_col (b_col_r),
                .c_val (c_val_r),
                .d_val (d_val[i][j])
            );
        `elsif TCU_BHF
            VX_tcu_fedp_bhf #(
                .LATENCY (FEDP_LATENCY),
                .N (TCU_TC_K)
            ) fedp (
                .clk   (clk),
                .reset (reset),
                .enable(fedp_enable),
                .fmt_s (fmt_s_r),
                .fmt_d (fmt_d_r),
                .a_row (a_row_r),
                .b_col (b_col_r),
                .c_val (c_val_r),
                .d_val (d_val[i][j])
            );
        `elsif TCU_DSP
            VX_tcu_fedp_dsp #(
                .LATENCY (FEDP_LATENCY),
                .N (TCU_TC_K)
            ) fedp (
                .clk   (clk),
                .reset (reset),
                .enable(fedp_enable),
                .fmt_s (fmt_s_r),
                .fmt_d (fmt_d_r),
                .a_row (a_row_r),
                .b_col (b_col_r),
                .c_val (c_val_r),
                .d_val (d_val[i][j])
            );
        `endif

        `ifdef DBG_TRACE_TCU
            always @(posedge clk) begin
                if (execute_if.valid && execute_if.ready) begin
                    `TRACE(3, ("%t: %s FEDP-enq: wid=%0d, i=%0d, j=%0d, m=%0d, n=%0d, a_row=", $time, INSTANCE_ID, execute_if.data.wid, i, j, step_m, step_n))
                    `TRACE_ARRAY1D(2, "0x%0h", a_row, TCU_TC_K)
                    `TRACE(3, (", b_col="));
                    `TRACE_ARRAY1D(2, "0x%0h", b_col, TCU_TC_K)
                    `TRACE(3, (", c_val=0x%0h (#%0d)\n", c_val, execute_if.data.uuid));
                end
                if (result_if.valid && result_if.ready) begin
                    `TRACE(3, ("%t: %s FEDP-deq: wid=%0d, i=%0d, j=%0d, d_val=0x%0h (#%0d)\n", $time, INSTANCE_ID, result_if.data.wid, i, j, d_val[i][j], result_if.data.uuid));
                end
            end
        `endif // DBG_TRACE_TCU
        end
    end

    // Result interface muxing: UMMA vs WMMA
    assign result_if.valid = umma_result_valid || wmma_result_valid;

    // UMMA result data
    wire [UUID_WIDTH-1:0] result_uuid = umma_result_valid ? umma_uuid : mdata_queue_dout[MDATA_WIDTH-1:MDATA_WIDTH-UUID_WIDTH];
    wire [NW_WIDTH-1:0] result_wid = umma_result_valid ? umma_wid : mdata_queue_dout[MDATA_WIDTH-UUID_WIDTH-1:MDATA_WIDTH-UUID_WIDTH-NW_WIDTH];
    wire [PC_BITS-1:0] result_pc = umma_result_valid ? umma_pc : mdata_queue_dout[MDATA_WIDTH-UUID_WIDTH-NW_WIDTH-1:MDATA_WIDTH-UUID_WIDTH-NW_WIDTH-PC_BITS];
    wire [NUM_REGS_BITS-1:0] result_rd = umma_result_valid ? '0 : mdata_queue_dout[NUM_REGS_BITS-1:0];

    assign result_if.data.wb = umma_result_valid ? 1'b0 : (wmma_result_valid && (|result_rd));
    assign result_if.data.tmask = {`NUM_THREADS{1'b1}};
    assign result_if.data.data = umma_result_valid ? '{default:32'b0} : d_val;
    assign result_if.data.pid = 0;
    assign result_if.data.sop = 1;
    assign result_if.data.eop = 1;
    assign result_if.data.rd = result_rd;
    assign result_if.data.uuid = result_uuid;
    assign result_if.data.wid = result_wid;
    assign result_if.data.PC = result_pc;

endmodule

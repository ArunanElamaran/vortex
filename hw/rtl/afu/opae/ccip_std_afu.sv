// Code reused from Intel OPAE's 04_local_memory sample program with changes made to fit Vortex

// Top Level Vortex Driver

// To be done:
// Check how to run this with OPAE. Looks like setup issue

`ifndef NOPAE

`include "platform_if.vh"

import local_mem_cfg_pkg::*;
import tensor_mem_cfg_pkg::*;

module ccip_std_afu #(
    parameter NUM_LOCAL_MEM_BANKS = 2,
    parameter NUM_TENSOR_MEM_BANKS = 2
) (
    // CCI-P Clocks and Resets
    input  logic        pClk,                 // Primary CCI-P interface clock.
    input  logic        pClkDiv2,             // Aligned, pClk divided by 2.
    input  logic        pClkDiv4,             // Aligned, pClk divided by 4.
    input  logic        uClk_usr,             // User clock domain. Refer to clock programming guide.
    input  logic        uClk_usrDiv2,         // Aligned, user clock divided by 2.
    input  logic        pck_cp2af_softReset,  // CCI-P ACTIVE HIGH Soft Reset

    input  logic [1:0]  pck_cp2af_pwrState,   // CCI-P AFU Power State
    input  logic        pck_cp2af_error,      // CCI-P Protocol Error Detected

    // CCI-P structures
    input  t_if_ccip_Rx pck_cp2af_sRx,        // CCI-P Rx Port
    output t_if_ccip_Tx pck_af2cp_sTx,        // CCI-P Tx Port

    // Local memory interface
    avalon_mem_if.to_fiu local_mem[NUM_LOCAL_MEM_BANKS],
    avalon_mem_if.to_fiu tensor_mem[NUM_TENSOR_MEM_BANKS]
);

    // ====================================================================
    // Pick the proper clk and reset, as chosen by the AFU's JSON file
    // ====================================================================

    // The platform may transform the CCI-P clock from pClk to a clock
    // chosen in the AFU's JSON file.
    logic clk;
    assign clk = `PLATFORM_PARAM_CCI_P_CLOCK;

    logic reset;
    assign reset = `PLATFORM_PARAM_CCI_P_RESET;


    // ====================================================================
    // Register signals at interface before consuming them
    // ====================================================================

    (* noprune *) logic [1:0]  cp2af_pwrState_T1;
    (* noprune *) logic        cp2af_error_T1;

    logic        reset_T1;
    t_if_ccip_Rx cp2af_sRx_T1;
    t_if_ccip_Tx af2cp_sTx_T0;

    ccip_interface_reg inst_green_ccip_interface_reg
       (
        .pClk                    (clk),
        .pck_cp2af_softReset_T0  (reset),
        .pck_cp2af_pwrState_T0   (pck_cp2af_pwrState),
        .pck_cp2af_error_T0      (pck_cp2af_error),
        .pck_cp2af_sRx_T0        (pck_cp2af_sRx),
        .pck_af2cp_sTx_T0        (af2cp_sTx_T0),

        .pck_cp2af_softReset_T1  (reset_T1),
        .pck_cp2af_pwrState_T1   (cp2af_pwrState_T1),
        .pck_cp2af_error_T1      (cp2af_error_T1),
        .pck_cp2af_sRx_T1        (cp2af_sRx_T1),
        .pck_af2cp_sTx_T1        (pck_af2cp_sTx)
        );


    // ====================================================================
    // User AFU goes here
    // ====================================================================

    t_local_mem_byte_mask avs_byteenableL [NUM_LOCAL_MEM_BANKS];
    logic                 avs_waitrequestL [NUM_LOCAL_MEM_BANKS];
    t_local_mem_data      avs_readdataL [NUM_LOCAL_MEM_BANKS];
    logic                 avs_readdatavalidL [NUM_LOCAL_MEM_BANKS];
    t_local_mem_burst_cnt avs_burstcountL [NUM_LOCAL_MEM_BANKS];
    t_local_mem_data      avs_writedataL [NUM_LOCAL_MEM_BANKS];
    t_local_mem_addr      avs_addressL [NUM_LOCAL_MEM_BANKS];
    logic                 avs_writeL [NUM_LOCAL_MEM_BANKS];
    logic                 avs_readL [NUM_LOCAL_MEM_BANKS];


    t_tensor_mem_byte_mask avs_byteenableT [NUM_TENSOR_MEM_BANKS];
    logic                 avs_waitrequestT [NUM_TENSOR_MEM_BANKS];
    t_tensor_mem_data      avs_readdataT [NUM_TENSOR_MEM_BANKS];
    logic                 avs_readdatavalidT [NUM_TENSOR_MEM_BANKS];
    t_tensor_mem_burst_cnt avs_burstcountT [NUM_TENSOR_MEM_BANKS];
    t_tensor_mem_data      avs_writedataT [NUM_TENSOR_MEM_BANKS];
    t_tensor_mem_addr      avs_addressT [NUM_TENSOR_MEM_BANKS];
    logic                 avs_writeT [NUM_TENSOR_MEM_BANKS];
    logic                 avs_readT [NUM_TENSOR_MEM_BANKS];

    for (genvar b = 0; b < NUM_LOCAL_MEM_BANKS; b++) begin
        assign local_mem[b].burstcount = avs_burstcountL[b];
        assign local_mem[b].writedata  = avs_writedataL[b];
        assign local_mem[b].address    = avs_addressL[b];
        assign local_mem[b].byteenable = avs_byteenableL[b];
        assign local_mem[b].write      = avs_writeL[b];
        assign local_mem[b].read       = avs_readL[b];

        assign avs_waitrequestL[b]   = local_mem[b].waitrequest;
        assign avs_readdataL[b]      = local_mem[b].readdata;
        assign avs_readdatavalidL[b] = local_mem[b].readdatavalid;
    end

    for (genvar b = 0; b < NUM_TENSOR_MEM_BANKS; b++) begin
        assign tensor_mem[b].burstcount = avs_burstcountT[b];
        assign tensor_mem[b].writedata  = avs_writedataT[b];
        assign tensor_mem[b].address    = avs_addressT[b];
        assign tensor_mem[b].byteenable = avs_byteenableT[b];
        assign tensor_mem[b].write      = avs_writeT[b];
        assign tensor_mem[b].read       = avs_readT[b];

        assign avs_waitrequestT[b]   = tensor_mem[b].waitrequest;
        assign avs_readdataT[b]      = tensor_mem[b].readdata;
        assign avs_readdatavalidT[b] = tensor_mem[b].readdatavalid;
    end

    vortex_afu #(
        .NUM_LOCAL_MEM_BANKS(NUM_LOCAL_MEM_BANKS)
    ) afu (
        .clk                 (clk),
        .reset               (reset_T1),

        .cp2af_sRxPort       (cp2af_sRx_T1),
        .af2cp_sTxPort       (af2cp_sTx_T0),

        .avs_writedataL       (avs_writedataL),
        .avs_readdataL        (avs_readdataL),
        .avs_addressL         (avs_addressL),
        .avs_waitrequestL     (avs_waitrequestL),
        .avs_writeL           (avs_writeL),
        .avs_readL            (avs_readL),
        .avs_byteenableL      (avs_byteenableL),
        .avs_burstcountL      (avs_burstcountL),
        .avs_readdatavalidL   (avs_readdatavalidL),


        .avs_writedataT       (avs_writedataT),
        .avs_readdataT        (avs_readdataT),
        .avs_addressT         (avs_addressT),
        .avs_waitrequestT     (avs_waitrequestT),
        .avs_writeT           (avs_writeT),
        .avs_readT            (avs_readT),
        .avs_byteenableT      (avs_byteenableT),
        .avs_burstcountT      (avs_burstcountT),
        .avs_readdatavalidT   (avs_readdatavalidT),
    );

endmodule

`endif

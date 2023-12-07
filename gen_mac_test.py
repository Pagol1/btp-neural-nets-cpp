# Generates Verilog Tests from CPP log file
import math

def pp(msg):
    global INDENT
    msg_len = len(msg)
    words = msg.split()

    if ("end" in words):
        INDENT -= 1
    elif ("end:" in words):
        INDENT -= 1
    if ("endgenerate" in words):
        INDENT -= 1

    op = "\t"*INDENT + msg + "\n"

    if ("begin" in words):
        INDENT += 1
    elif ("begin:" in words):
        INDENT += 1
    if ("generate" in words):
        INDENT += 1
        
    assert(INDENT >= 0)
    return op

def gen_mac_test(in_path):
    global INDENT
    
    vectors = []
    with open(in_path, "r") as f:
        for line in f:
            vectors.append(line.split())

    op = ""
    op += pp("module fxp32_mac_tb;")
    INDENT += 1
    op += pp("// Inputs")
    op += pp("reg clk;")
    op += pp("reg rstn;")
    op += pp("reg prstn;")
    op += pp("reg acc;")
    op += pp("reg [31:0] in_a;")
    op += pp("reg [31:0] in_b;")
    op += "\n"
    op += pp("// Outputs")
    op += pp("wire [31:0] out_c;")
    op += "\n"
    op += pp("fxp32_mac uut(.clk(clk), .rstn(rstn), .prstn(prstn), .acc(acc), .in_a(in_a), .in_b(in_b), .out_c(out_c));")
    op += "\n"
    op += pp("initial begin")
    op += pp("clk <= 0; rstn <= 0; acc <= 0; prstn <= 0; in_a <= 0; in_b <= 0;")
    op += pp("// Wait 100 ns for global reset to finish")
    op += pp("#100;")
    op += pp("rstn <= 1'b1; prstn <= 1'b1; acc <= 1'b1; ")
    for vec in vectors:
        s = "#50; "
        for ele in vec:
            sig, val = ele.split(":")
            pre, act = val.split("'")
            if sig == "acc":
                s += " //"
            s += sig + " <= " + pre + "'h" + act[1:].upper() + "; "
        op += pp(s)
    op += pp("end")
    op += "\n"
    op += pp("initial begin")
    op += pp("#100; clk <= 0;")
    op += pp("forever begin")
    op += pp("#25 clk <= ~clk;")
    op += pp("end")
    op += pp("end")
    INDENT -= 1
    op += pp("endmodule")

    return op

if __name__ == "__main__":
    INDENT = 0
    op = gen_mac_test("./build/cpp_test_vec.txt")
    with open("./build/fxp32_mac_tb.v", "w") as f:
        f.write(op)

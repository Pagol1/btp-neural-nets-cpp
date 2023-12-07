#include "test_mac.h"

/* ToDO
 * [] Identify sign and shit
 * [] Get printing function down
 * [] Make random runs of the MAC
 * [] Print to file
 * [] Parse in python
 */

void print_fixed(fixed_test a)
{
    std::bitset<BIT_LEN> x(a.data_);
    std::cout << x << "\n" ;
}

void print_fixed_format(std::ofstream &outf, fixed_test a)
{
    std::bitset<BIT_LEN> x(a.data_);
    outf << "32'h" <<  std::hex << std::setfill('0') << std::setw(8) << x.to_ulong() << " " ;
}

void MAC(fixed_test in_a, fixed_test in_b, fixed_test &acc) {
    acc += in_a * in_b;
}

void log_MAC(std::ofstream &outf, fixed_test in_a,  fixed_test in_b, fixed_test acc) {
    outf << "in_a:"; print_fixed_format(outf, in_a);
    outf << "in_b:"; print_fixed_format(outf, in_b);
    outf << "acc:"; print_fixed_format(outf, acc); outf << "\n";
}

void compare_fixed(std::string a, std::string b) {
    fixed_test A, B;
    A.data_ = std::stoll(a, 0, 16);
    B.data_ = std::stoll(b, 0, 16);
    std::cout << A << " " << B << "\n";
}

int main()
{
    /*
    compare_fixed("9f6fc234", "60903dcb");
    compare_fixed("6e254b8b", "e0903dca");
    compare_fixed("2f2cc39d", "1fa191b9");
    compare_fixed("dec4f88f", "2c6865cc");
    compare_fixed("a96ac34f", "a0b7ab5c");
    compare_fixed("c3a6f2ff", "ea3a048f");
    */
    
    int NUM_TESTS = 10;
    fixed_test in_a, in_b, acc;
    acc = 0;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dist(0, 0.5);

    std::ofstream outf{"cpp_test_vec.txt"};
    if (!outf) {
        std::cerr << "Could not open file for writing\n";
        return EXIT_FAILURE;
    }

    for (int i=0; i<NUM_TESTS; ++i) {
        in_a = dist(gen); in_b = dist(gen);
        log_MAC(outf, in_a, in_b, acc);
        MAC(in_a, in_b, acc);
    }
    
    return EXIT_SUCCESS;
}

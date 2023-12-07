#include "MNIST.h"
#include "my_defines.h"
#include <iterator>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <arpa/inet.h>
#include <time.h>
#include <random>

/*
class CSVRow
{
    public:
        std::string_view operator[](std::size_t index) const
        {
            return std::string_view(&m_line[m_data[index] + 1], m_data[index + 1] -  (m_data[index] + 1));
        }
        std::size_t size() const
        {
            return m_data.size() - 1;
        }
        void readNextRow(std::istream& str)
        {
            std::getline(str, m_line);

            m_data.clear();
            m_data.emplace_back(-1);
            std::string::size_type pos = 0;
            while((pos = m_line.find(',', pos)) != std::string::npos)
            {
                m_data.emplace_back(pos);
                ++pos;
            }
            // This checks for a trailing comma with no data after it.
            pos   = m_line.size();
            m_data.emplace_back(pos);
        }
    private:
        std::string         m_line;
        std::vector<int>    m_data;
};

class CSVIterator
{   
    public:
        typedef std::input_iterator_tag     iterator_category;
        typedef CSVRow                      value_type;
        typedef std::size_t                 difference_type;
        typedef CSVRow*                     pointer;
        typedef CSVRow&                     reference;

        CSVIterator(std::istream& str)  :m_str(str.good()?&str:nullptr) { ++(*this); }
        CSVIterator()                   :m_str(nullptr) {}

        // Pre Increment
        CSVIterator& operator++()               {if (m_str) { if (!((*m_str) >> m_row)){m_str = nullptr;}}return *this;}
        // Post increment
        CSVIterator operator++(int)             {CSVIterator    tmp(*this);++(*this);return tmp;}
        CSVRow const& operator*()   const       {return m_row;}
        CSVRow const* operator->()  const       {return &m_row;}

        bool operator==(CSVIterator const& rhs) {return ((this == &rhs) || ((this->m_str == nullptr) && (rhs.m_str == nullptr)));}
        bool operator!=(CSVIterator const& rhs) {return !((*this) == rhs);}
    private:
        std::istream*       m_str;
        CSVRow              m_row;
};

class CSVRange
{
    std::istream&   stream;
    public:
        CSVRange(std::istream& str)
            : stream(str)
        {}
        CSVIterator begin() const {return CSVIterator{stream};}
        CSVIterator end()   const {return CSVIterator{};}
};
*/

bool load_data(std::string fname, std::vector<std::vector<uint8_t>> &data)
{
    std::ifstream file(fname, std::ios::binary);
    if (!file) {
        std::cerr << "Cannot open the file! " << fname << std::endl;
        return 1;
    }
    
    uint32_t magicNumber;
    uint32_t numItems;
    uint32_t numRows, numCols;
    // Read the header
    ;
    file.read(reinterpret_cast<char *>(&magicNumber), sizeof(magicNumber));
    file.read(reinterpret_cast<char *>(&numItems), sizeof(numItems));
    file.read(reinterpret_cast<char *>(&numRows), sizeof(numRows));
    file.read(reinterpret_cast<char *>(&numCols), sizeof(numCols));

    // Convert magic number to MSB first (if on a little-endian machine)
    uint32_t magicNum = ntohl(magicNumber);  // Requires #include <arpa/inet.h> on some platforms
    uint32_t numberOfItems = ntohl(numItems); // Assume it's in the correct byte order
    uint32_t rows = ntohl(numRows);
    uint32_t cols = ntohl(numCols);

    std::cout << "Magic Number: 0x" << std::hex << magicNum << "(" << std::dec << magicNum << ")" << std::endl;
    std::cout << "Number of items: " << numberOfItems << std::endl;

    // Read labels
    data.resize(numberOfItems);
    for (auto &d : data) d.resize(rows*cols);
    for (size_t j=0; j<numberOfItems; ++j) {
        file.read(reinterpret_cast<char*>(data[j].data()), rows*cols);
    }
    file.close();
    return true;
}


bool load_labels(std::string fname, std::vector<uint8_t> &labels)
{
    std::ifstream file(fname, std::ios::binary);
    if (!file) {
        std::cerr << "Cannot open the file! " << fname <<  std::endl;
        return false;
    }
    
    uint32_t magicNumber;
    uint32_t numItems;
    // Read the header
    ;
    file.read(reinterpret_cast<char *>(&magicNumber), sizeof(magicNumber));
    file.read(reinterpret_cast<char *>(&numItems), sizeof(numItems));
    // Convert magic number to MSB first (if on a little-endian machine)
    uint32_t magicNum = ntohl(magicNumber);  // Requires #include <arpa/inet.h> on some platforms
    uint32_t numberOfItems = ntohl(numItems); // Assume it's in the correct byte order

    std::cout << "Magic Number: 0x" << std::hex << magicNum << "(" << std::dec << magicNum << ")" << std::endl;
    std::cout << "Number of items: " << numberOfItems << std::endl;

    // Read labels
    labels.resize(numberOfItems);
    file.read(reinterpret_cast<char*>(labels.data()), numberOfItems);
    file.close();
    return true;
}


int main() {
    srand(time(NULL));
    std::vector<std::vector<uint8_t>> train_x, test_x;
    std::vector<uint8_t> train_y, test_y;
    load_data("../data/train-images-idx3-ubyte", train_x); 
    load_labels("../data/train-labels-idx1-ubyte", train_y); 
    load_data("../data/t10k-images-idx3-ubyte", test_x); 
    load_labels("../data/t10k-labels-idx1-ubyte", test_y);
    std::cout << std::endl;
    std::cout << static_cast<fixed>(1) << " " << static_cast<fixed>(-0.5)  << std::endl;
    MNIST mnist(train_x, train_y, test_x, test_y);
    for (int epoch=1; epoch<=100; ++epoch) {
        std::cout << "======================================== EPOCH " << epoch << " ==============================================\n";
        mnist.train();
        mnist.test();
    }
    return 0;
}

#include <iostream>
#include <regex>
#include <string>
#include <vector>
#include <sstream>
#include <stdexcept>
// ostream_iterator
#include <iterator>

#define uchar unsigned char

std::vector<uchar> hexToBytes(const std::string &hexString) {
    // if (hexString.length() % 2 != 0) {
    //     throw std::runtime_error("Invalid hex string length for conversion to bytes.");
    // }
    
    size_t len = hexString.length() / 2;
    std::vector<uchar> bytes;
    bytes.reserve(len);

    for (size_t i = 0; i < len; ++i) {
        std::string byteString = hexString.substr(2 * i, 2);
        uchar byte = std::stoi(byteString, nullptr, 16);
        bytes.push_back(byte);
    }

    return bytes;
}

std::vector<uchar> findPythonByteObjects(const std::string &input) {
    // Regular expression pattern to match Python byte literals b'...'
    std::regex byteObjectPattern(R"(b'([^']*)')");

    std::smatch match;
    std::vector<uchar> byteArray;

    std::string::const_iterator searchStart = input.cbegin();
    while (std::regex_search(searchStart, input.cend(), match, byteObjectPattern)) {
        // Extract the contents of the byte literal (safe to assume ASCII - no UTF8 handling)
        std::string byteString = match[1];

        // Replace escape sequences with actual byte values
        std::regex escapePattern(R"(\\x([a-fA-F0-9]{2}))");
        std::ostringstream replacedByteString;
        std::regex_replace(std::ostream_iterator<uchar>(replacedByteString),
                           byteString.begin(), byteString.end(),
                           escapePattern, "$1");

        // Convert replaced byte string to byte array
        byteArray = hexToBytes(replacedByteString.str());

        // In the original function, only the first match is used - hence the break
        break;

        searchStart = match.suffix().first;
    }

    return byteArray;
}

// int main() {
//     std::string input = "Some text with Python byte object b'\\x61\\x62\\x63'";
    
//     try {
//         std::vector<uchar> byteArray = findPythonByteObjects(input);
//         std::cout << "Byte array: [ ";
//         for (uchar byte : byteArray) {
//             std::cout << std::hex << static_cast<int>(byte) << " ";
//         }
//         std::cout << "]" << std::endl;
//     } catch (const std::runtime_error &e) {
//         std::cerr << "Error: " << e.what() << std::endl;
//     }
    
//     return 0;
// }
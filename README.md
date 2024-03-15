# One Billion Row Challenge (1BRC)
Link: https://github.com/gunnarmorling/1brc/

The One Billion Row Challenge (1BRC) is a challenge that aims to aggregate one billion rows from a text file using various optimization techniques. The implementation in the provided repository showcases a solution using Modern C++.

### Comparison with other implementations (On My Machine):
| Implementation                                                            |   Time  |
|---------------------------------------------------------------------------|:-------:|
| [lehuyduc](https://github.com/lehuyduc/1brc-simd)                         |  5.575s |
| [tooomasy](https://github.com/tooomasy/1brc-cpp)                          |  8.413s |
| [kajott](https://gist.github.com/kajott/8a7c3f56a9a035fdfe3b2c79e17c8eed) | 10.580s |

### Key features of the implementation include:
- SIMD instructions for efficient parallel processing.
- Custom hash table with linear probing for fast data lookup.
- Memory mapping technique for efficient file reading.
- Multi-threaded processing to leverage parallelism.
- Usage of std::string_view instead of std::string for improved performance.
- Parsing temperature values as integers instead of floats.

### How to run:
`mkdir build; cd build; cmake .. && make && ./1brc_cpp <input-filename> <output-result-filename>`

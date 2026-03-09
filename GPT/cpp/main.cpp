#include<iostream>
#include<fstream>
#include<vector>
#include<string>
#include<algorithm>
#include "value.h"
#include "tensor.h"
#include "model.h"

int main(){
    std::ifstream file("input.txt");

    if(!file.is_open())
    {
        std::cerr << "input.txt 파일을 찾을 수 없습니다!" << std::endl;
        return 1;
    }

    std::vector<std::string> docs;
    std::string line;
    std::string all_chars = "";

    while(std::getline(file, line))
    {
        if(!line.empty()){
            if(!line.empty()){
                docs.push_back(line);
                all_chars += line;
            }
        }
    }

    std::sort(all_chars.begin(), all_chars.end());
    auto last = std::unique(all_chars.begin(), all_chars.end());
    all_chars.erase(last, all_chars.end());

    int BOS = all_chars.size();
    int VOCAB_SIZE = all_chars.size() + 1 ; 

    std::cout << "Num docs: " << docs.size() << std::endl;
    std::cout << "Vocab size: " << VOCAB_SIZE << std::endl;


    return 0;
}
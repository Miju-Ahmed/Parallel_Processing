/*
Command for run
!nvcc -arch=sm_75 Miju.cu -o miju
!time ./miju AKASH 50 > output.txt
*/

%%writefile Miju.cu
#include <bits/stdc++.h>
#include <cuda.h>
using namespace std;

struct Contact {
    char name[65];
    char phone_number[65];
};

// Utility: Trim whitespace
string trim(const string& str) {
    size_t start = str.find_first_not_of(" \t");
    if (start == string::npos) return "";
    size_t end = str.find_last_not_of(" \t");
    return str.substr(start, end - start + 1);
}

// Extract name between two //
string getInput(ifstream& file) {
    string line;
    getline(file, line);

    size_t first_delim = line.find("//");
    if (first_delim == string::npos) return "";

    size_t second_delim = line.find("//", first_delim + 2);
    if (second_delim == string::npos) return "";

    return trim(line.substr(first_delim + 2, second_delim - (first_delim + 2)));
}

// Extract phone number after last //
string getPhoneNumber(ifstream& file) {
    string line;
    getline(file, line);

    size_t delim = line.rfind("//");
    if (delim == string::npos) return "";

    return trim(line.substr(delim + 2));
}

__device__ bool check(char* str1, char* str2) {
    for (int i = 0; str1[i] != '\0'; i++) {
        int flag = 1;
        for (int j = 0; str2[j] != '\0'; j++) {
            if (str1[i + j] == '\0' || str1[i + j] != str2[j]) {
                flag = 0;
                break;
            }
        }
        if (flag == 1) return true;
    }
    return false;
}

__global__ void myKernel(Contact* phoneBook, char* pat, int offset, Contact* matches, int* matchCount) {
    int threadNumber = threadIdx.x + offset;
    if (check(phoneBook[threadNumber].name, pat)) {
        int idx = atomicAdd(matchCount, 1);
        matches[idx] = phoneBook[threadNumber];
    }
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        cerr << "Usage: " << argv[0] << " <search_name> <thread_limit>" << endl;
        return 1;
    }

    int threadLimit = atoi(argv[2]);
    string search_name = argv[1];

    ifstream myfile("/content/drive/MyDrive/labtest_dataset.txt");
    if (!myfile.is_open()) {
        cerr << "Error opening file!" << endl;
        return 1;
    }

    vector<Contact> phoneBook;
    string line;
    int count = 0;

    while (getline(myfile, line)) {
        if (line.empty()) continue;
        if (++count > 200000) break;

        string name = getInput(myfile);
        string phoneNum = getPhoneNumber(myfile);

        Contact c;
        strncpy(c.name, name.c_str(), sizeof(c.name) - 1);
        c.name[sizeof(c.name) - 1] = '\0';
        strncpy(c.phone_number, phoneNum.c_str(), sizeof(c.phone_number) - 1);
        c.phone_number[sizeof(c.phone_number) - 1] = '\0';

        phoneBook.push_back(c);
    }

    char pat[65];
    strncpy(pat, search_name.c_str(), sizeof(pat) - 1);
    pat[sizeof(pat) - 1] = '\0';

    // Allocate memory on device
    int n = phoneBook.size();
    Contact* d_phoneBook;
    char* d_pat;
    Contact* d_matches;
    int* d_matchCount;

    cudaMalloc(&d_phoneBook, n * sizeof(Contact));
    cudaMemcpy(d_phoneBook, phoneBook.data(), n * sizeof(Contact), cudaMemcpyHostToDevice);

    cudaMalloc(&d_pat, 65);
    cudaMemcpy(d_pat, pat, 65, cudaMemcpyHostToDevice);

    cudaMalloc(&d_matches, n * sizeof(Contact)); // Worst case: all match
    cudaMalloc(&d_matchCount, sizeof(int));
    cudaMemset(d_matchCount, 0, sizeof(int));

    int remaining = n;
    int offset = 0;
    while (remaining > 0) {
        int batchSize = min(threadLimit, remaining);
        myKernel<<<1, batchSize>>>(d_phoneBook, d_pat, offset, d_matches, d_matchCount);
        cudaDeviceSynchronize();
        remaining -= batchSize;
        offset += batchSize;
    }

    // Get match count
    int h_matchCount;
    cudaMemcpy(&h_matchCount, d_matchCount, sizeof(int), cudaMemcpyDeviceToHost);

    // Copy matches back
    vector<Contact> matches(h_matchCount);
    cudaMemcpy(matches.data(), d_matches, h_matchCount * sizeof(Contact), cudaMemcpyDeviceToHost);

    // Sort matches by name
    sort(matches.begin(), matches.end(), [](const Contact& a, const Contact& b) {
        return strcmp(a.name, b.name) < 0;
    });

    // Print sorted matches
    for (const auto& c : matches) {
        cout << c.name << " " << c.phone_number << endl;
    }

    // Cleanup
    cudaFree(d_phoneBook);
    cudaFree(d_pat);
    cudaFree(d_matches);
    cudaFree(d_matchCount);

    return 0;
}

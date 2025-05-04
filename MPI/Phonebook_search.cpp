/*
    How to run:
    mpic++ -o search phonebook_search.cpp
    mpirun -np 4 ./search phonebook1.txt Bob
*/



//Simplified
#include <bits/stdc++.h>
#include <mpi.h>
using namespace std;

struct Contact {
    string name;
    string phone;
};

void send_string(const string &text, int receiver) {
    int len = text.size() + 1;
    MPI_Send(&len, 1, MPI_INT, receiver, 1, MPI_COMM_WORLD);
    MPI_Send(text.c_str(), len, MPI_CHAR, receiver, 1, MPI_COMM_WORLD);
}

string receive_string(int sender) {
    int len;
    MPI_Recv(&len, 1, MPI_INT, sender, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    char *buf = new char[len];
    MPI_Recv(buf, len, MPI_CHAR, sender, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    string res(buf);
    delete[] buf;
    return res;
}

string vector_to_string(const vector<Contact> &contacts, int start, int end) {
    string result;
    for (int i = start; i < min((int)contacts.size(), end); i++) {
        result += contacts[i].name + "," + contacts[i].phone + "\n";
    }
    return result;
}

vector<Contact> string_to_contacts(const string &text) {
    vector<Contact> contacts;
    istringstream iss(text);
    string line;
    while (getline(iss, line)) {
        if (line.empty()) continue;
        int comma = line.find(",");
        if (comma == string::npos) continue;
        contacts.push_back({line.substr(0, comma), line.substr(comma + 1)});
    }
    return contacts;
}

string check(const Contact &c, const string &search) {
    if (c.name.find(search) != string::npos) {
        return c.name + " " + c.phone + "\n";
    }
    return "";
}

void read_phonebook(const vector<string> &files, vector<Contact> &contacts) {
    for (const string &file : files) {
        ifstream f(file);
        string line;
        while (getline(f, line)) {
            if (line.empty()) continue;
            int comma = line.find(",");
            if (comma == string::npos) continue;
            contacts.push_back({line.substr(1, comma - 2), line.substr(comma + 2, line.size() - comma - 3)});
        }
    }
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 3) {
        if (rank == 0)
            cerr << "Usage: mpirun -n <procs> " << argv[0] << " <file>... <search>\n";
        MPI_Finalize();
        return 1;
    }

    string search_term = argv[argc - 1];
    double start, end;

    if (rank == 0) {
        vector<string> files(argv + 1, argv + argc - 1);
        vector<Contact> contacts;
        read_phonebook(files, contacts);
        int total = contacts.size();
        int chunk = (total + size - 1) / size;

        for (int i = 1; i < size; i++) {
            string text = vector_to_string(contacts, i * chunk, (i + 1) * chunk);
            send_string(text, i);
        }

        start = MPI_Wtime();
        string result;
        for (int i = 0; i < min(chunk, total); i++) {
            string match = check(contacts[i], search_term);
            if (!match.empty()) result += match;
        }
        end = MPI_Wtime();

        for (int i = 1; i < size; i++) {
            string recv = receive_string(i);
            if (!recv.empty()) result += recv;
        }
        
        ofstream out("output.txt");
        out << result;
        out.close();
        // printf("%s \n", result.c_str());
        printf("Process %d took %f seconds.\n", rank, end - start);

    } else {
        string recv_text = receive_string(0);
        vector<Contact> contacts = string_to_contacts(recv_text);
        start = MPI_Wtime();
        string result;
        for (auto &c : contacts) {
            string match = check(c, search_term);
            if (!match.empty()) result += match;
        }
        end = MPI_Wtime();
        send_string(result, 0);
        printf("Process %d took %f seconds.\n", rank, end - start);
    }

    MPI_Finalize();
    return 0;
}

/*
#include <bits/stdc++.h>
#include <mpi.h>

using namespace std;

// Send a string to a specified receiver.
void send_string(const string &text, int receiver) {
    int length = text.size() + 1;  // include terminating null
    MPI_Send(&length, 1, MPI_INT, receiver, 1, MPI_COMM_WORLD);
    MPI_Send(text.c_str(), length, MPI_CHAR, receiver, 1, MPI_COMM_WORLD);
}

// Receive a string from a specified sender.
string receive_string(int sender) {
    int length;
    MPI_Recv(&length, 1, MPI_INT, sender, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    char *text = new char[length];
    MPI_Recv(text, length, MPI_CHAR, sender, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    string result(text);
    delete[] text;
    return result;
}

// Convert a vector of strings to a single string with newline separators.
string vector_to_string(const vector<string> &words, int start, int end) {
    string text;
    for (int i = start; i < min((int)words.size(), end); i++) {
        text += words[i] + "\n";
    }
    return text;
}

// Convert a newline-separated string into a vector of strings.
vector<string> string_to_vector(const string &text) {
    vector<string> words;
    istringstream iss(text);
    string line;
    while(getline(iss, line)) {
        if (!line.empty())
            words.push_back(line);
    }
    return words;
}

// Performs an exact match check. If the name exactly matches search_name,
// returns a formatted string containing the name, phone, and the process rank.
// Otherwise returns an empty string.

// Exact matching
// string check(const string &name, const string &phone, const string &search_name, int rank) {
//     if (name == search_name) {
//         return name + " " + phone + " found by process " + to_string(rank) + "\n";
//     }
//     return "";
// }

// Partial matching (e.g., "Bob" should match with "Bobby")
string check(const string &name, const string &phone, const string &search_name, int world_rank) {
    if (name.find(search_name) != string::npos) {
        return name + " " + phone + "\n";
    }
    return "";
}


// Reads the phonebook files. Each file should contain lines with a name and a phone number.
void read_phonebook(const vector<string> &file_names, vector<string> &names, vector<string> &phone_numbers) {
    for (const auto &file_name : file_names) {
        ifstream file(file_name);
        if (!file) {
            cerr << "Error opening file: " << file_name << "\n";
            continue;
        }
        string name, number;
        while (file >> name >> number) {
            names.push_back(name);
            phone_numbers.push_back(number);
        }
        file.close();
    }
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Check for proper usage: at least one phonebook file and a search term.
    if (argc < 3) {
        if (world_rank == 0)
            cerr << "Usage: mpirun -n <num_procs> " << argv[0] << " <phonebook_file> ... <search_term>\n";
        MPI_Finalize();
        return 1;
    }

    // The last command-line argument is the search term.
    string search = argv[argc - 1];
    double start_time, finish_time;

    if (world_rank == 0) {
        // Root reads the phonebook files.
        vector<string> names, phone_numbers;
        vector<string> file_names(argv + 1, argv + argc - 1);
        read_phonebook(file_names, names, phone_numbers);

        int total_contacts = names.size();
        // Compute a roughly equal chunk size for each process.
        int chunk = (total_contacts + world_size - 1) / world_size;

        // Send each non-root process its assigned chunk.
        for (int proc = 1; proc < world_size; proc++) {
            int start = proc * chunk;
            int end = start + chunk;
            string names_to_send = vector_to_string(names, start, end);
            send_string(names_to_send, proc);
            string phones_to_send = vector_to_string(phone_numbers, start, end);
            send_string(phones_to_send, proc);
            send_string(search, proc);
        }

        // Process 0 works on the first chunk.
        string local_results;
        start_time = MPI_Wtime();
        int end = min(chunk, total_contacts);
        for (int i = 0; i < end; i++) {
            string res = check(names[i], phone_numbers[i], search, world_rank);
            if (!res.empty())
                local_results += res;
        }
        finish_time = MPI_Wtime();

        // Gather search results from other processes.
        for (int proc = 1; proc < world_size; proc++) {
            string received = receive_string(proc);
            if (!received.empty())
                local_results += received;
        }

        // Instead of printing, you can now send local_results to some other destination.
        // Here we print the final results at the root.
        // cout << "\nSearch Results:\n" << local_results;
        printf("Process %d took %f seconds.\n", world_rank, finish_time - start_time);
    } else {
        // Non-root processes: receive the assigned chunk from the root.
        string received_names = receive_string(0);
        vector<string> names = string_to_vector(received_names);
        string received_phones = receive_string(0);
        vector<string> phone_numbers = string_to_vector(received_phones);
        search = receive_string(0);

        string local_results;
        start_time = MPI_Wtime();
        for (size_t i = 0; i < names.size(); i++) {
            string res = check(names[i], phone_numbers[i], search, world_rank);
            if (!res.empty())
                local_results += res;
        }
        finish_time = MPI_Wtime();

        // Send local search results back to the root.
        send_string(local_results, 0);
        printf("Process %d took %f seconds.\n", world_rank, finish_time - start_time);
    }

    MPI_Finalize();
    return 0;
}
*/
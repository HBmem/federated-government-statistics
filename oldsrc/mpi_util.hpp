#pragma once
#include <mpi.h>
#include <string>
#include <vector>

inline std::string bcastString(int root, MPI_Comm comm, std::string s) {
    int rank;
    MPI_Comm_rank(comm, &rank);

    int length = (rank == root) ? (int)s.size() : 0;
    MPI_Bcast(&length, 1, MPI_INT, root, comm);

    if (rank != root) {
        s.resize(length);
    }
    if (length > 0) {
        MPI_Bcast(s.data(), length, MPI_CHAR, root, comm);
    }
    return s;
}

inline std::vector<std::string> gatherStringsToRoot(int root, MPI_Comm comm, const std::string &local) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int local_length = (int)local.size();
    std::vector<int> lengths(size, 0);
    MPI_Gather(&local_length, 1, MPI_INT, lengths.data(), 1, MPI_INT, root, comm);

    std::vector<int> displs(size, 0);
    int total_length = 0;

    if (rank == root) {
        for (int i = 0; i < size; i++) {
            displs[i] = total_length;
            total_length += lengths[i];
        }
    }

    std::vector<char> recv_buffer(total_length);
    if (rank == root) {
        recv_buffer.resize(total_length);
    }

    MPI_Gatherv(local.data(), local_length, MPI_CHAR, recv_buffer.data(), lengths.data(), displs.data(), MPI_CHAR, root, comm);
    
    std::vector<std::string> out;
    if (rank == root) {
        out.resize(size);
        for (int i = 0; i < size; i++) {
            out[i] = std::string(recv_buffer.data() + displs[i], lengths[i]);
        }
    }
    return out;
}
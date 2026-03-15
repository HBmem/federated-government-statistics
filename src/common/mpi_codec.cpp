#include "mpi_codec.hpp"

#include <vector>

namespace mpi_codec {

void send_string(int dest_rank, int tag, const std::string& s, MPI_Comm comm) {
  int len = static_cast<int>(s.size());
  MPI_Send(&len, 1, MPI_INT, dest_rank, tag, comm);
  if (len > 0) {
    MPI_Send(s.data(), len, MPI_CHAR, dest_rank, tag, comm);
  }
}

std::string recv_string(int src_rank, int tag, MPI_Comm comm) {
  MPI_Status st{};
  int len = 0;
  MPI_Recv(&len, 1, MPI_INT, src_rank, tag, comm, &st);

  std::string out;
  out.resize(static_cast<size_t>(len));

  if (len > 0) {
    MPI_Recv(out.data(), len, MPI_CHAR, src_rank, tag, comm, &st);
  }
  return out;
}

bool try_recv_string_any(int& out_src_rank, int tag, std::string& out_payload, MPI_Comm comm) {
  MPI_Status st{};
  int flag = 0;

  // Probe for incoming message length.
  MPI_Iprobe(MPI_ANY_SOURCE, tag, comm, &flag, &st);
  if (!flag) return false;

  out_src_rank = st.MPI_SOURCE;

  int len = 0;
  MPI_Recv(&len, 1, MPI_INT, out_src_rank, tag, comm, &st);

  std::string out;
  out.resize(static_cast<size_t>(len));
  if (len > 0) {
    MPI_Recv(out.data(), len, MPI_CHAR, out_src_rank, tag, comm, &st);
  }

  out_payload = std::move(out);
  return true;
}

} // namespace mpi_codec
#pragma once

#include <mpi.h>
#include <string>

/*
  MPI codec for sending JSON payloads as length-prefixed strings.

  We use two MPI_Send calls:
  1) send int length
  2) send bytes
*/
namespace mpi_codec {

constexpr int TAG_CONTROL = 100;  // coordinator -> worker control messages
constexpr int TAG_DATA    = 101;  // worker -> coordinator responses

void send_string(int dest_rank, int tag, const std::string& s, MPI_Comm comm);
std::string recv_string(int src_rank, int tag, MPI_Comm comm);

bool try_recv_string_any(int& out_src_rank, int tag, std::string& out_payload, MPI_Comm comm);

}
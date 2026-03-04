#pragma once
#include "db.hpp"

enum class SchemaType {
    A,
    B,
    C,
    Unknown
};

SchemaType detectSchema(PgDb &db);
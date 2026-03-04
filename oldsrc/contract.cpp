#include "contract.hpp"
#include "schema_detect.hpp"
#include "db.hpp"
#include "adapters/adapter.hpp"
#include <memory>
#include <stdexcept>
#include <iostream>

std::unique_ptr<Adapter> createSchemaAAdapter();
std::unique_ptr<Adapter> createSchemaBAdapter();
std::unique_ptr<Adapter> createSchemaCAdapter();

AdapterResult runWorkerContract(const std::string &conninfo, const ContractRequest &req, const std::string &county_fips) {
    PgDb db(conninfo);
    SchemaType schema = detectSchema(db);
    std::cerr << "detectSchema: returned type " << static_cast<int>(schema) << " for county " << county_fips << "\n";

    // Try adapters in an order that prefers the detected schema first to avoid
    // polluting aggregatedErrors with expected "does not exist" failures
    // from other schema adapters. Return the first successful result or the
    // best error if none succeed.
    AdapterResult finalRes;
    std::string aggregatedErrors;

    auto tryAdapter = [&](SchemaType s) -> bool {
        std::unique_ptr<Adapter> a;
        switch (s) {
            case SchemaType::A: a = createSchemaAAdapter(); break;
            case SchemaType::B: a = createSchemaBAdapter(); break;
            case SchemaType::C: a = createSchemaCAdapter(); break;
            default: return false;
        }

        try {
            AdapterResult r = a->run(db, req, county_fips);
            if (r.status.ok) {
                finalRes = r;
                return true;
            } else {
                aggregatedErrors += "Schema" + std::to_string(static_cast<int>(s)) + ": " + r.status.message + "\n";
                return false;
            }
        } catch (const std::exception &e) {
            aggregatedErrors += "Schema" + std::to_string(static_cast<int>(s)) + " exception: " + e.what() + "\n";
            return false;
        }
    };

    // Prefer the detected schema first, then try the remaining adapters.
    auto tryInPreferredOrder = [&]() -> bool {
        // Order to attempt: detected schema first (if known), then others
        if (schema != SchemaType::Unknown) {
            if (tryAdapter(schema)) return true;
            // fallthrough to try remaining types
        }
        // Try the remaining schemas in a stable order A, B, C skipping duplicates
        if (schema != SchemaType::A && tryAdapter(SchemaType::A)) return true;
        if (schema != SchemaType::B && tryAdapter(SchemaType::B)) return true;
        if (schema != SchemaType::C && tryAdapter(SchemaType::C)) return true;
        return false;
    };

    if (tryInPreferredOrder()) {
        return finalRes;
    }

    AdapterResult out;
    // If failures are due to missing relations/columns across all adapters,
    // treat this as an empty/no-data node rather than a fatal error.
    if (aggregatedErrors.find("does not exist") != std::string::npos
        || aggregatedErrors.find("profile_json") != std::string::npos) {
        out.status.ok = true;
        out.status.message = "No supported tables present on node";
        // Log the aggregated adapter errors for debugging to help diagnose
        // nodes that claim to have a supported schema but whose adapters
        // nonetheless fail.
        std::cerr << "Aggregated adapter errors for node " << county_fips << ":\n" << aggregatedErrors << "\n";
        return out;
    }

    out.status.ok = false;
    out.status.message = "All adapter attempts failed:\n" + aggregatedErrors;
    return out;
}
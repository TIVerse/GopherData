// Package json provides JSON reading and writing functionality for DataFrames.
//
// Supports two orientations:
//   - "records": [{"col": val, ...}, ...] (default)
//   - "columns": {"col": [val, ...], ...}
//
// Also supports JSONL format (one record per line).
//
// Example:
//
//	// Read JSON
//	df, err := json.ReadJSON("data.json",
//	    json.Orient("records"),
//	)
//
//	// Write JSON
//	err = json.WriteJSON(df, "output.json",
//	    json.Orient("columns"),
//	)
//
//	// JSONL format
//	df, err := json.ReadJSON("data.jsonl", json.Lines())
package json

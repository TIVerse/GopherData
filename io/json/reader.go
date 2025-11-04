// Package json provides JSON reading and writing functionality for DataFrames.
package json

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"

	"github.com/TIVerse/GopherData/dataframe"
)

// JSONReader reads JSON files into DataFrames.
type JSONReader struct {
	path   string
	orient string // "records" or "columns"
	lines  bool   // JSONL format (one record per line)
}

// JSONOption is a functional option for configuring JSONReader.
type JSONOption func(*JSONReader) error

// Orient sets the JSON format orientation.
// "records": [{"col": val, ...}, ...]
// "columns": {"col": [val, ...], ...}
func Orient(format string) JSONOption {
	return func(r *JSONReader) error {
		if format != "records" && format != "columns" {
			return fmt.Errorf("invalid orient %q: must be 'records' or 'columns'", format)
		}
		r.orient = format
		return nil
	}
}

// Lines enables JSONL format (one record per line).
func Lines() JSONOption {
	return func(r *JSONReader) error {
		r.lines = true
		return nil
	}
}

// ReadJSON reads a JSON file and returns a DataFrame.
func ReadJSON(path string, opts ...JSONOption) (*dataframe.DataFrame, error) {
	reader := &JSONReader{
		path:   path,
		orient: "records", // Default
		lines:  false,
	}

	// Apply options
	for _, opt := range opts {
		if err := opt(reader); err != nil {
			return nil, err
		}
	}

	return reader.read()
}

// read performs the actual JSON reading.
func (r *JSONReader) read() (*dataframe.DataFrame, error) {
	file, err := os.Open(r.path)
	if err != nil {
		return nil, fmt.Errorf("failed to open file: %w", err)
	}
	defer func() { _ = file.Close() }()

	if r.lines {
		return r.readJSONLines(file)
	}

	if r.orient == "records" {
		return r.readRecords(file)
	}

	return r.readColumns(file)
}

// readRecords reads JSON in records format: [{"col": val, ...}, ...]
func (r *JSONReader) readRecords(file *os.File) (*dataframe.DataFrame, error) {
	var records []map[string]any
	
	decoder := json.NewDecoder(file)
	if err := decoder.Decode(&records); err != nil {
		return nil, fmt.Errorf("failed to decode JSON: %w", err)
	}

	if len(records) == 0 {
		return dataframe.New(map[string]any{})
	}

	return dataframe.FromRecords(records)
}

// readColumns reads JSON in columns format: {"col": [val, ...], ...}
func (r *JSONReader) readColumns(file *os.File) (*dataframe.DataFrame, error) {
	var data map[string][]any
	
	decoder := json.NewDecoder(file)
	if err := decoder.Decode(&data); err != nil {
		return nil, fmt.Errorf("failed to decode JSON: %w", err)
	}

	// Convert []any to any (which will be handled by New())
	convertedData := make(map[string]any, len(data))
	for key, val := range data {
		convertedData[key] = val
	}

	return dataframe.New(convertedData)
}

// readJSONLines reads JSONL format (one record per line).
func (r *JSONReader) readJSONLines(file *os.File) (*dataframe.DataFrame, error) {
	var records []map[string]any
	
	scanner := bufio.NewScanner(file)
	
	for scanner.Scan() {
		line := scanner.Bytes()
		if len(line) == 0 {
			continue
		}

		var record map[string]any
		if err := json.Unmarshal(line, &record); err != nil {
			return nil, fmt.Errorf("failed to decode JSON line: %w", err)
		}

		records = append(records, record)
	}

	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("error reading file: %w", err)
	}

	if len(records) == 0 {
		return dataframe.New(map[string]any{})
	}

	return dataframe.FromRecords(records)
}

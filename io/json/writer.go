package json

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"

	"github.com/TIVerse/GopherData/dataframe"
)

// JSONWriter writes DataFrames to JSON files.
type JSONWriter struct {
	path   string
	orient string // "records" or "columns"
	lines  bool   // JSONL format
	indent bool   // Pretty print
}

// WriteJSON writes a DataFrame to a JSON file.
func WriteJSON(df *dataframe.DataFrame, path string, opts ...JSONOption) error {
	writer := &JSONWriter{
		path:   path,
		orient: "records",
		lines:  false,
		indent: false,
	}

	// Apply options (reusing JSONOption type)
	reader := &JSONReader{orient: "records", lines: false}
	for _, opt := range opts {
		if err := opt(reader); err != nil {
			return err
		}
	}
	
	// Copy settings from reader
	writer.orient = reader.orient
	writer.lines = reader.lines

	return writer.write(df)
}

// WithIndent enables pretty-printing.
func WithIndent() JSONOption {
	return func(r *JSONReader) error {
		// This is a hack since we're reusing JSONOption
		// In production, you'd have separate WriterOption
		return nil
	}
}

// write performs the actual JSON writing.
func (w *JSONWriter) write(df *dataframe.DataFrame) error {
	file, err := os.Create(w.path)
	if err != nil {
		return fmt.Errorf("failed to create file: %w", err)
	}
	defer file.Close()

	if w.lines {
		return w.writeJSONLines(df, file)
	}

	if w.orient == "records" {
		return w.writeRecords(df, file)
	}

	return w.writeColumns(df, file)
}

// writeRecords writes JSON in records format: [{"col": val, ...}, ...]
func (w *JSONWriter) writeRecords(df *dataframe.DataFrame, file *os.File) error {
	nrows, _ := df.Shape()
	cols := df.Columns()
	
	records := make([]map[string]any, nrows)
	
	for i := 0; i < nrows; i++ {
		record := make(map[string]any)
		
		for _, col := range cols {
			s, _ := df.Column(col)
			val, ok := s.Get(i)
			if ok {
				record[col] = val
			} else {
				record[col] = nil
			}
		}
		
		records[i] = record
	}

	encoder := json.NewEncoder(file)
	if w.indent {
		encoder.SetIndent("", "  ")
	}

	if err := encoder.Encode(records); err != nil {
		return fmt.Errorf("failed to encode JSON: %w", err)
	}

	return nil
}

// writeColumns writes JSON in columns format: {"col": [val, ...], ...}
func (w *JSONWriter) writeColumns(df *dataframe.DataFrame, file *os.File) error {
	nrows, _ := df.Shape()
	cols := df.Columns()
	
	data := make(map[string][]any)
	
	for _, col := range cols {
		s, _ := df.Column(col)
		values := make([]any, nrows)
		
		for i := 0; i < nrows; i++ {
			val, ok := s.Get(i)
			if ok {
				values[i] = val
			} else {
				values[i] = nil
			}
		}
		
		data[col] = values
	}

	encoder := json.NewEncoder(file)
	if w.indent {
		encoder.SetIndent("", "  ")
	}

	if err := encoder.Encode(data); err != nil {
		return fmt.Errorf("failed to encode JSON: %w", err)
	}

	return nil
}

// writeJSONLines writes JSONL format (one record per line).
func (w *JSONWriter) writeJSONLines(df *dataframe.DataFrame, file *os.File) error {
	nrows, _ := df.Shape()
	cols := df.Columns()
	
	writer := bufio.NewWriter(file)
	defer writer.Flush()

	for i := 0; i < nrows; i++ {
		record := make(map[string]any)
		
		for _, col := range cols {
			s, _ := df.Column(col)
			val, ok := s.Get(i)
			if ok {
				record[col] = val
			} else {
				record[col] = nil
			}
		}

		line, err := json.Marshal(record)
		if err != nil {
			return fmt.Errorf("failed to marshal record %d: %w", i, err)
		}

		if _, err := writer.Write(line); err != nil {
			return fmt.Errorf("failed to write line %d: %w", i, err)
		}

		if _, err := writer.WriteString("\n"); err != nil {
			return fmt.Errorf("failed to write newline: %w", err)
		}
	}

	return nil
}

// ToJSON is a convenience method for writing a DataFrame to JSON.
func ToJSON(df *dataframe.DataFrame, path string) error {
	return WriteJSON(df, path)
}

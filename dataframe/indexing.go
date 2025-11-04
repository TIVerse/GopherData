package dataframe

import (
	"fmt"
	"time"

	"github.com/TIVerse/GopherData/core"
)

// RangeIndex is an integer-based index with start, stop, and step.
// Represents indices [start, start+step, start+2*step, ..., stop-1].
type RangeIndex struct {
	start int
	stop  int
	step  int
}

// NewRangeIndex creates a new RangeIndex.
func NewRangeIndex(start, stop, step int) *RangeIndex {
	if step == 0 {
		step = 1
	}
	return &RangeIndex{start: start, stop: stop, step: step}
}

// Len returns the number of elements in the index.
func (ri *RangeIndex) Len() int {
	if ri.step > 0 {
		if ri.stop <= ri.start {
			return 0
		}
		return (ri.stop - ri.start + ri.step - 1) / ri.step
	}
	if ri.start <= ri.stop {
		return 0
	}
	return (ri.start - ri.stop - ri.step - 1) / (-ri.step)
}

// Get returns the label at the given position.
func (ri *RangeIndex) Get(pos int) any {
	if pos < 0 || pos >= ri.Len() {
		return nil
	}
	return ri.start + pos*ri.step
}

// Slice returns a subset of the index.
func (ri *RangeIndex) Slice(start, end int) core.Index {
	if start < 0 {
		start = 0
	}
	if end > ri.Len() {
		end = ri.Len()
	}
	if start >= end {
		return NewRangeIndex(0, 0, 1)
	}

	newStart := ri.start + start*ri.step
	newStop := ri.start + end*ri.step

	return NewRangeIndex(newStart, newStop, ri.step)
}

// Loc returns the integer positions for the given labels.
func (ri *RangeIndex) Loc(labels ...any) ([]int, error) {
	positions := make([]int, 0, len(labels))

	for _, label := range labels {
		// Convert label to int
		var val int
		switch v := label.(type) {
		case int:
			val = v
		case int64:
			val = int(v)
		case int32:
			val = int(v)
		default:
			return nil, fmt.Errorf("label %v: expected int, got %T: %w", label, label, core.ErrKeyNotFound)
		}

		// Check if value is in range
		if ri.step > 0 {
			if val < ri.start || val >= ri.stop {
				return nil, fmt.Errorf("label %d: %w", val, core.ErrKeyNotFound)
			}
			if (val-ri.start)%ri.step != 0 {
				return nil, fmt.Errorf("label %d: %w", val, core.ErrKeyNotFound)
			}
			pos := (val - ri.start) / ri.step
			positions = append(positions, pos)
		} else {
			if val > ri.start || val <= ri.stop {
				return nil, fmt.Errorf("label %d: %w", val, core.ErrKeyNotFound)
			}
			if (ri.start-val)%(-ri.step) != 0 {
				return nil, fmt.Errorf("label %d: %w", val, core.ErrKeyNotFound)
			}
			pos := (ri.start - val) / (-ri.step)
			positions = append(positions, pos)
		}
	}

	return positions, nil
}

// Copy returns a copy of the index.
func (ri *RangeIndex) Copy() core.Index {
	return &RangeIndex{start: ri.start, stop: ri.stop, step: ri.step}
}

// StringIndex is a string-based index with a lookup map for fast label-based access.
type StringIndex struct {
	labels []string
	lookup map[string]int
}

// NewStringIndex creates a new StringIndex.
func NewStringIndex(labels []string) *StringIndex {
	lookup := make(map[string]int, len(labels))
	for i, label := range labels {
		lookup[label] = i
	}
	return &StringIndex{labels: labels, lookup: lookup}
}

// Len returns the number of elements in the index.
func (si *StringIndex) Len() int {
	return len(si.labels)
}

// Get returns the label at the given position.
func (si *StringIndex) Get(pos int) any {
	if pos < 0 || pos >= len(si.labels) {
		return nil
	}
	return si.labels[pos]
}

// Slice returns a subset of the index.
func (si *StringIndex) Slice(start, end int) core.Index {
	if start < 0 {
		start = 0
	}
	if end > len(si.labels) {
		end = len(si.labels)
	}
	if start >= end {
		return NewStringIndex([]string{})
	}

	newLabels := make([]string, end-start)
	copy(newLabels, si.labels[start:end])

	return NewStringIndex(newLabels)
}

// Loc returns the integer positions for the given labels.
func (si *StringIndex) Loc(labels ...any) ([]int, error) {
	positions := make([]int, 0, len(labels))

	for _, label := range labels {
		str, ok := label.(string)
		if !ok {
			return nil, fmt.Errorf("label %v: expected string, got %T: %w", label, label, core.ErrKeyNotFound)
		}

		pos, exists := si.lookup[str]
		if !exists {
			return nil, fmt.Errorf("label %q: %w", str, core.ErrKeyNotFound)
		}

		positions = append(positions, pos)
	}

	return positions, nil
}

// Copy returns a copy of the index.
func (si *StringIndex) Copy() core.Index {
	newLabels := make([]string, len(si.labels))
	copy(newLabels, si.labels)
	return NewStringIndex(newLabels)
}

// DatetimeIndex is a time-based index for time series data.
type DatetimeIndex struct {
	times []time.Time
	tz    *time.Location
}

// NewDatetimeIndex creates a new DatetimeIndex.
func NewDatetimeIndex(times []time.Time, tz *time.Location) *DatetimeIndex {
	if tz == nil {
		tz = time.UTC
	}
	return &DatetimeIndex{times: times, tz: tz}
}

// Len returns the number of elements in the index.
func (di *DatetimeIndex) Len() int {
	return len(di.times)
}

// Get returns the label at the given position.
func (di *DatetimeIndex) Get(pos int) any {
	if pos < 0 || pos >= len(di.times) {
		return nil
	}
	return di.times[pos]
}

// Slice returns a subset of the index.
func (di *DatetimeIndex) Slice(start, end int) core.Index {
	if start < 0 {
		start = 0
	}
	if end > len(di.times) {
		end = len(di.times)
	}
	if start >= end {
		return NewDatetimeIndex([]time.Time{}, di.tz)
	}

	newTimes := make([]time.Time, end-start)
	copy(newTimes, di.times[start:end])

	return NewDatetimeIndex(newTimes, di.tz)
}

// Loc returns the integer positions for the given labels.
func (di *DatetimeIndex) Loc(labels ...any) ([]int, error) {
	positions := make([]int, 0, len(labels))

	for _, label := range labels {
		t, ok := label.(time.Time)
		if !ok {
			return nil, fmt.Errorf("label %v: expected time.Time, got %T: %w", label, label, core.ErrKeyNotFound)
		}

		// Linear search for matching time
		found := false
		for i, dt := range di.times {
			if dt.Equal(t) {
				positions = append(positions, i)
				found = true
				break
			}
		}

		if !found {
			return nil, fmt.Errorf("label %v: %w", t, core.ErrKeyNotFound)
		}
	}

	return positions, nil
}

// Copy returns a copy of the index.
func (di *DatetimeIndex) Copy() core.Index {
	newTimes := make([]time.Time, len(di.times))
	copy(newTimes, di.times)
	return NewDatetimeIndex(newTimes, di.tz)
}

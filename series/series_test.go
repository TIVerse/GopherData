package series

import (
	"testing"

	"github.com/TIVerse/GopherData/core"
)

func TestSeriesNew(t *testing.T) {
	data := []int64{1, 2, 3, 4, 5}
	s := New("test", data, core.DtypeInt64)

	if s.Name() != "test" {
		t.Errorf("Expected name 'test', got %s", s.Name())
	}

	if s.Len() != 5 {
		t.Errorf("Expected length 5, got %d", s.Len())
	}

	if s.Dtype() != core.DtypeInt64 {
		t.Errorf("Expected dtype Int64, got %s", s.Dtype())
	}
}

func TestSeriesGetSet(t *testing.T) {
	data := []int64{10, 20, 30}
	s := New("test", data, core.DtypeInt64)

	// Get valid value
	val, ok := s.Get(1)
	if !ok {
		t.Error("Expected Get to return ok=true for valid index")
	}
	if val != int64(20) {
		t.Errorf("Expected value 20, got %v", val)
	}

	// Set new value
	err := s.Set(1, int64(25))
	if err != nil {
		t.Errorf("Set failed: %v", err)
	}

	val, ok = s.Get(1)
	if !ok || val != int64(25) {
		t.Errorf("Expected value 25 after Set, got %v", val)
	}
}

func TestSeriesNullHandling(t *testing.T) {
	data := []int64{10, 20, 30, 40, 50}
	s := New("test", data, core.DtypeInt64)

	// Mark index 2 as null
	s.SetNull(2)

	if !s.IsNull(2) {
		t.Error("Expected index 2 to be null")
	}

	if s.NullCount() != 1 {
		t.Errorf("Expected null count 1, got %d", s.NullCount())
	}

	// Get should return false for null
	_, ok := s.Get(2)
	if ok {
		t.Error("Expected Get to return ok=false for null value")
	}
}

func TestSeriesFillNA(t *testing.T) {
	data := []int64{10, 20, 30, 40}
	s := New("test", data, core.DtypeInt64)
	s.SetNull(1)
	s.SetNull(3)

	filled := s.FillNA(int64(99))

	// Check original is unchanged
	if s.NullCount() != 2 {
		t.Error("Original series should still have nulls")
	}

	// Check filled series
	if filled.NullCount() != 0 {
		t.Error("Filled series should have no nulls")
	}

	val, ok := filled.Get(1)
	if !ok || val != int64(99) {
		t.Errorf("Expected filled value 99, got %v", val)
	}
}

func TestSeriesDropNA(t *testing.T) {
	data := []int64{10, 20, 30, 40, 50}
	s := New("test", data, core.DtypeInt64)
	s.SetNull(1)
	s.SetNull(3)

	dropped := s.DropNA()

	if dropped.Len() != 3 {
		t.Errorf("Expected length 3 after dropping nulls, got %d", dropped.Len())
	}

	if dropped.NullCount() != 0 {
		t.Error("Dropped series should have no nulls")
	}
}

func TestSeriesNumericOperations(t *testing.T) {
	data := []int64{10, 20, 30, 40, 50}
	s := New("test", data, core.DtypeInt64)

	sum := Sum(s)
	if sum != 150 {
		t.Errorf("Expected sum 150, got %d", sum)
	}

	mean := Mean(s)
	if mean != 30.0 {
		t.Errorf("Expected mean 30.0, got %f", mean)
	}

	min, ok := Min(s)
	if !ok || min != 10 {
		t.Errorf("Expected min 10, got %d", min)
	}

	max, ok := Max(s)
	if !ok || max != 50 {
		t.Errorf("Expected max 50, got %d", max)
	}
}

func TestSeriesNumericWithNulls(t *testing.T) {
	data := []int64{10, 20, 30, 40, 50}
	s := New("test", data, core.DtypeInt64)
	s.SetNull(1) // Null at 20
	s.SetNull(3) // Null at 40

	// Operations should skip nulls
	sum := Sum(s)
	if sum != 90 {
		t.Errorf("Expected sum 90 (10+30+50), got %d", sum)
	}

	mean := Mean(s)
	if mean != 30.0 {
		t.Errorf("Expected mean 30.0 (90/3), got %f", mean)
	}
}

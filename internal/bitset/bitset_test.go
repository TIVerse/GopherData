package bitset

import "testing"

func TestBitSetBasic(t *testing.T) {
	bs := New(100)

	if bs.Len() != 100 {
		t.Errorf("Expected length 100, got %d", bs.Len())
	}

	// Test Set and Test
	bs.Set(42)
	if !bs.Test(42) {
		t.Error("Expected bit 42 to be set")
	}

	// Test Clear
	bs.Clear(42)
	if bs.Test(42) {
		t.Error("Expected bit 42 to be cleared")
	}

	// Test Count
	bs.Set(0)
	bs.Set(10)
	bs.Set(50)
	bs.Set(99)
	if count := bs.Count(); count != 4 {
		t.Errorf("Expected count 4, got %d", count)
	}
}

func TestBitSetClone(t *testing.T) {
	bs := New(50)
	bs.Set(10)
	bs.Set(20)
	bs.Set(30)

	clone := bs.Clone()

	if clone.Len() != bs.Len() {
		t.Error("Clone length mismatch")
	}

	if clone.Count() != bs.Count() {
		t.Error("Clone count mismatch")
	}

	// Modify original, clone should not change
	bs.Set(40)
	if clone.Test(40) {
		t.Error("Clone was modified when original changed")
	}
}

func TestBitSetSlice(t *testing.T) {
	bs := New(100)
	bs.Set(10)
	bs.Set(20)
	bs.Set(30)
	bs.Set(50)

	sliced := bs.Slice(15, 35)

	if sliced.Len() != 20 {
		t.Errorf("Expected sliced length 20, got %d", sliced.Len())
	}

	// Bit 20 (original) should be at position 5 in slice (20-15=5)
	if !sliced.Test(5) {
		t.Error("Expected bit 5 to be set in sliced bitset")
	}

	// Bit 30 (original) should be at position 15 in slice (30-15=15)
	if !sliced.Test(15) {
		t.Error("Expected bit 15 to be set in sliced bitset")
	}
}

func TestBitSetAnyNoneAll(t *testing.T) {
	bs := New(10)

	if bs.Any() {
		t.Error("Expected Any() to be false for empty bitset")
	}

	if !bs.None() {
		t.Error("Expected None() to be true for empty bitset")
	}

	bs.Set(5)

	if !bs.Any() {
		t.Error("Expected Any() to be true after setting a bit")
	}

	if bs.None() {
		t.Error("Expected None() to be false after setting a bit")
	}

	bs.SetAll()

	if !bs.All() {
		t.Error("Expected All() to be true after SetAll()")
	}
}

// Benchmark for Set operation
func BenchmarkBitSetSet(b *testing.B) {
	bs := New(10000000)
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		bs.Set(i % 10000000)
	}
}

// Benchmark for Test operation
func BenchmarkBitSetTest(b *testing.B) {
	bs := New(10000000)
	for i := 0; i < 10000000; i += 2 {
		bs.Set(i)
	}
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_ = bs.Test(i % 10000000)
	}
}

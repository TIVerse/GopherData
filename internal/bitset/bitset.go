// Package bitset provides a bit-packed set implementation for efficient null masks.
package bitset

// BitSet is a bit-packed array using uint64 words.
// Each bit represents a boolean value (1 = set/true, 0 = clear/false).
// Used primarily for null masks in Series where 1 = null.
type BitSet struct {
	bits []uint64
	len  int
}

const (
	wordSize  = 64
	wordShift = 6  // log2(64)
	wordMask  = 63 // 64 - 1
)

// New creates a new BitSet with the specified length.
// All bits are initialized to 0 (cleared).
func New(n int) *BitSet {
	if n < 0 {
		n = 0
	}
	numWords := (n + wordSize - 1) / wordSize
	return &BitSet{
		bits: make([]uint64, numWords),
		len:  n,
	}
}

// Len returns the number of bits in the set.
func (bs *BitSet) Len() int {
	return bs.len
}

// Set sets the bit at position i to 1.
// Panics if i is out of bounds.
func (bs *BitSet) Set(i int) {
	if i < 0 || i >= bs.len {
		panic("bitset: index out of bounds")
	}
	wordIdx := i >> wordShift
	bitIdx := uint(i & wordMask)
	bs.bits[wordIdx] |= 1 << bitIdx
}

// Clear sets the bit at position i to 0.
// Panics if i is out of bounds.
func (bs *BitSet) Clear(i int) {
	if i < 0 || i >= bs.len {
		panic("bitset: index out of bounds")
	}
	wordIdx := i >> wordShift
	bitIdx := uint(i & wordMask)
	bs.bits[wordIdx] &^= 1 << bitIdx
}

// Test returns true if the bit at position i is set (1).
// Panics if i is out of bounds.
func (bs *BitSet) Test(i int) bool {
	if i < 0 || i >= bs.len {
		panic("bitset: index out of bounds")
	}
	wordIdx := i >> wordShift
	bitIdx := uint(i & wordMask)
	return (bs.bits[wordIdx] & (1 << bitIdx)) != 0
}

// Count returns the number of set bits (population count).
func (bs *BitSet) Count() int {
	count := 0
	for _, word := range bs.bits {
		count += popcount(word)
	}
	return count
}

// Clone returns a deep copy of the BitSet.
func (bs *BitSet) Clone() *BitSet {
	newBits := make([]uint64, len(bs.bits))
	copy(newBits, bs.bits)
	return &BitSet{
		bits: newBits,
		len:  bs.len,
	}
}

// ClearAll sets all bits to 0.
func (bs *BitSet) ClearAll() {
	for i := range bs.bits {
		bs.bits[i] = 0
	}
}

// SetAll sets all bits to 1.
func (bs *BitSet) SetAll() {
	for i := range bs.bits {
		bs.bits[i] = ^uint64(0)
	}
	// Clear excess bits in the last word
	if bs.len%wordSize != 0 {
		lastWordBits := bs.len % wordSize
		bs.bits[len(bs.bits)-1] &= (1 << uint(lastWordBits)) - 1
	}
}

// Any returns true if any bit is set.
func (bs *BitSet) Any() bool {
	for _, word := range bs.bits {
		if word != 0 {
			return true
		}
	}
	return false
}

// None returns true if no bits are set.
func (bs *BitSet) None() bool {
	return !bs.Any()
}

// All returns true if all bits are set.
func (bs *BitSet) All() bool {
	return bs.Count() == bs.len
}

// popcount returns the number of set bits in x (population count).
// Uses the MIT HAKMEM algorithm.
func popcount(x uint64) int {
	// Alternative: use bits.OnesCount64 from math/bits (Go 1.9+)
	x -= (x >> 1) & 0x5555555555555555
	x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333)
	x = (x + (x >> 4)) & 0x0f0f0f0f0f0f0f0f
	return int((x * 0x0101010101010101) >> 56)
}

// Slice returns a new BitSet containing bits from start (inclusive) to end (exclusive).
func (bs *BitSet) Slice(start, end int) *BitSet {
	if start < 0 {
		start = 0
	}
	if end > bs.len {
		end = bs.len
	}
	if start >= end {
		return New(0)
	}

	newLen := end - start
	result := New(newLen)

	for i := 0; i < newLen; i++ {
		if bs.Test(start + i) {
			result.Set(i)
		}
	}

	return result
}

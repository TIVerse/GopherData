// Package bitset provides a space-efficient bit array implementation.
//
// BitSet uses 1 bit per value (packed into uint64 words) rather than 1 byte,
// providing an 8x memory savings compared to []bool.
//
// Primary use case: null masks in Series where each bit indicates if a value is null.
// Target performance: <10ns per Set/Test operation.
package bitset

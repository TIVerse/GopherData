package core

import "testing"

func TestDtypeString(t *testing.T) {
	tests := []struct {
		dtype    Dtype
		expected string
	}{
		{DtypeInt64, "int64"},
		{DtypeFloat64, "float64"},
		{DtypeString, "string"},
		{DtypeBool, "bool"},
		{DtypeTime, "datetime"},
		{DtypeCategory, "category"},
	}

	for _, tt := range tests {
		if got := tt.dtype.String(); got != tt.expected {
			t.Errorf("Dtype.String() = %v, want %v", got, tt.expected)
		}
	}
}

func TestOrderString(t *testing.T) {
	tests := []struct {
		order    Order
		expected string
	}{
		{Ascending, "ascending"},
		{Descending, "descending"},
	}

	for _, tt := range tests {
		if got := tt.order.String(); got != tt.expected {
			t.Errorf("Order.String() = %v, want %v", got, tt.expected)
		}
	}
}

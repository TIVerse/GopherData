package stats

import (
	"math"
	"testing"
)

func TestMean(t *testing.T) {
	tests := []struct {
		name     string
		values   []float64
		expected float64
	}{
		{"Simple", []float64{1, 2, 3, 4, 5}, 3.0},
		{"WithDecimals", []float64{1.5, 2.5, 3.5}, 2.5},
		{"Empty", []float64{}, 0.0},
		{"SingleValue", []float64{5.0}, 5.0},
	}
	
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := Mean(tt.values)
			if math.Abs(result-tt.expected) > 1e-9 {
				t.Errorf("Mean() = %v, want %v", result, tt.expected)
			}
		})
	}
}

func TestMedian(t *testing.T) {
	tests := []struct {
		name     string
		values   []float64
		expected float64
	}{
		{"OddLength", []float64{1, 3, 2, 5, 4}, 3.0},
		{"EvenLength", []float64{1, 2, 3, 4}, 2.5},
		{"SingleValue", []float64{5.0}, 5.0},
		{"Empty", []float64{}, 0.0},
	}
	
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := Median(tt.values)
			if math.Abs(result-tt.expected) > 1e-9 {
				t.Errorf("Median() = %v, want %v", result, tt.expected)
			}
		})
	}
}

func TestMode(t *testing.T) {
	values := []float64{1, 2, 2, 3, 3, 3, 4}
	mode, freq := Mode(values)
	
	if mode != 3.0 {
		t.Errorf("Mode() = %v, want 3.0", mode)
	}
	if freq != 3 {
		t.Errorf("Mode frequency = %v, want 3", freq)
	}
}

func TestStd(t *testing.T) {
	values := []float64{2, 4, 4, 4, 5, 5, 7, 9}
	result := Std(values)
	expected := 2.138 // Approximate sample std
	
	if math.Abs(result-expected) > 0.01 {
		t.Errorf("Std() = %v, want approximately %v", result, expected)
	}
}

func TestQuantile(t *testing.T) {
	values := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
	
	tests := []struct {
		q        float64
		expected float64
	}{
		{0.0, 1.0},
		{0.25, 3.25},
		{0.5, 5.5},
		{0.75, 7.75},
		{1.0, 10.0},
	}
	
	for _, tt := range tests {
		result := Quantile(values, tt.q)
		if math.Abs(result-tt.expected) > 0.1 {
			t.Errorf("Quantile(%v) = %v, want %v", tt.q, result, tt.expected)
		}
	}
}

func TestDescribe(t *testing.T) {
	values := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
	stats := Describe(values)
	
	if stats.Count != 10 {
		t.Errorf("Count = %v, want 10", stats.Count)
	}
	
	if math.Abs(stats.Mean-5.5) > 0.01 {
		t.Errorf("Mean = %v, want 5.5", stats.Mean)
	}
	
	if math.Abs(stats.Median-5.5) > 0.01 {
		t.Errorf("Median = %v, want 5.5", stats.Median)
	}
	
	if stats.Min != 1.0 {
		t.Errorf("Min = %v, want 1.0", stats.Min)
	}
	
	if stats.Max != 10.0 {
		t.Errorf("Max = %v, want 10.0", stats.Max)
	}
}

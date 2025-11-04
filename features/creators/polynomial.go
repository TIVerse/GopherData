// Package creators provides feature creation transformers.
package creators

import (
	"fmt"

	"github.com/TIVerse/GopherData/core"
	"github.com/TIVerse/GopherData/dataframe"
	seriesPkg "github.com/TIVerse/GopherData/series"
)

// PolynomialFeatures generates polynomial and interaction features.
// For degree=2 and columns [a, b]: creates [1, a, b, a², a×b, b²]
type PolynomialFeatures struct {
	// Degree of the polynomial features
	Degree int
	
	// IncludeBias adds a constant feature (all 1s)
	IncludeBias bool
	
	// InteractionOnly only creates interaction features (no powers)
	InteractionOnly bool
	
	// Columns to use. If nil, use all numeric columns.
	Columns []string
	
	fitted bool
}

// NewPolynomialFeatures creates a new PolynomialFeatures transformer.
func NewPolynomialFeatures(degree int) *PolynomialFeatures {
	return &PolynomialFeatures{
		Degree:          degree,
		IncludeBias:     true,
		InteractionOnly: false,
		fitted:          false,
	}
}

// Fit is a no-op for PolynomialFeatures (stateless transformer).
func (p *PolynomialFeatures) Fit(df *dataframe.DataFrame, _ ...string) error {
	cols := p.Columns
	if cols == nil {
		cols = getNumericColumns(df)
	}
	
	if len(cols) == 0 {
		return fmt.Errorf("no numeric columns to create features from")
	}
	
	p.Columns = cols
	p.fitted = true
	return nil
}

// Transform creates polynomial and interaction features.
func (p *PolynomialFeatures) Transform(df *dataframe.DataFrame) (*dataframe.DataFrame, error) {
	if !p.fitted {
		return nil, fmt.Errorf("transformer not fitted")
	}
	
	result := df.Copy()
	
	// Add bias term
	if p.IncludeBias {
		bias := make([]any, df.Nrows())
		for i := range bias {
			bias[i] = float64(1.0)
		}
		result = result.WithColumn("bias", seriesPkg.New("bias", bias, core.DtypeFloat64))
	}
	
	// Generate polynomial features
	if p.Degree >= 2 {
		result = p.addPolynomialFeatures(result, p.Degree)
	}
	
	return result, nil
}

// addPolynomialFeatures adds polynomial and interaction terms.
func (p *PolynomialFeatures) addPolynomialFeatures(df *dataframe.DataFrame, degree int) *dataframe.DataFrame {
	result := df
	
	if degree >= 2 {
		result = p.addDegreeFeatures(result, df, 2)
	}
	
	// For degree > 2, recursively add higher order terms
	if degree > 2 {
		for d := 3; d <= degree; d++ {
			result = p.addDegreeFeatures(result, df, d)
		}
	}
	
	return result
}

// addDegreeFeatures adds polynomial features of a specific degree
func (p *PolynomialFeatures) addDegreeFeatures(result *dataframe.DataFrame, df *dataframe.DataFrame, degree int) *dataframe.DataFrame {
	if degree == 2 {
		// Add squared terms (if not interaction only)
		if !p.InteractionOnly {
			for _, col := range p.Columns {
				colSeries, _ := df.Column(col)
				squared := make([]any, colSeries.Len())
				
				for i := 0; i < colSeries.Len(); i++ {
					val, ok := colSeries.Get(i)
					if !ok {
						squared[i] = nil
						continue
					}
					floatVal := toFloat64Creator(val)
					squared[i] = floatVal * floatVal
				}
				
				newColName := fmt.Sprintf("%s^2", col)
				result = result.WithColumn(newColName, seriesPkg.New(newColName, squared, core.DtypeFloat64))
			}
		}
		
		// Add pairwise interactions
		for i, col1 := range p.Columns {
			for j := i + 1; j < len(p.Columns); j++ {
				col2 := p.Columns[j]
				
				colSeries1, _ := df.Column(col1)
				colSeries2, _ := df.Column(col2)
				
				interaction := make([]any, colSeries1.Len())
				for k := 0; k < colSeries1.Len(); k++ {
					val1, ok1 := colSeries1.Get(k)
					val2, ok2 := colSeries2.Get(k)
					
					if !ok1 || !ok2 {
						interaction[k] = nil
						continue
					}
					
					interaction[k] = toFloat64Creator(val1) * toFloat64Creator(val2)
				}
				
				newColName := fmt.Sprintf("%s*%s", col1, col2)
				result = result.WithColumn(newColName, seriesPkg.New(newColName, interaction, core.DtypeFloat64))
			}
		}
	} else if degree > 2 {
		// For higher degrees, create powers and interactions
		if !p.InteractionOnly {
			// Add pure power terms (x^degree)
			for _, col := range p.Columns {
				colSeries, _ := df.Column(col)
				powered := make([]any, colSeries.Len())
				
				for i := 0; i < colSeries.Len(); i++ {
					val, ok := colSeries.Get(i)
					if !ok {
						powered[i] = nil
						continue
					}
					floatVal := toFloat64Creator(val)
					poweredVal := 1.0
					for d := 0; d < degree; d++ {
						poweredVal *= floatVal
					}
					powered[i] = poweredVal
				}
				
				newColName := fmt.Sprintf("%s^%d", col, degree)
				result = result.WithColumn(newColName, seriesPkg.New(newColName, powered, core.DtypeFloat64))
			}
		}
		
		// Add interaction terms for this degree
		// Generate all combinations that multiply to the given degree
		result = p.addInteractionTerms(result, df, degree)
	}
	
	return result
}

// addInteractionTerms adds interaction terms for a specific degree
func (p *PolynomialFeatures) addInteractionTerms(result *dataframe.DataFrame, df *dataframe.DataFrame, degree int) *dataframe.DataFrame {
	// For simplicity, add terms like x1*x2^(degree-1), x1^2*x2^(degree-2), etc.
	// This is a simplified version that creates major interaction terms
	
	if degree == 3 {
		// Add x1*x2*x3 type interactions
		n := len(p.Columns)
		for i := 0; i < n; i++ {
			for j := i + 1; j < n; j++ {
				for k := j + 1; k < n; k++ {
					col1, col2, col3 := p.Columns[i], p.Columns[j], p.Columns[k]
					
					series1, _ := df.Column(col1)
					series2, _ := df.Column(col2)
					series3, _ := df.Column(col3)
					
					interaction := make([]any, series1.Len())
					for idx := 0; idx < series1.Len(); idx++ {
						v1, ok1 := series1.Get(idx)
						v2, ok2 := series2.Get(idx)
						v3, ok3 := series3.Get(idx)
						
						if !ok1 || !ok2 || !ok3 {
							interaction[idx] = nil
							continue
						}
						
						interaction[idx] = toFloat64Creator(v1) * toFloat64Creator(v2) * toFloat64Creator(v3)
					}
					
					newColName := fmt.Sprintf("%s*%s*%s", col1, col2, col3)
					result = result.WithColumn(newColName, seriesPkg.New(newColName, interaction, core.DtypeFloat64))
				}
			}
		}
		
		// Add x1^2*x2 type interactions
		for i := 0; i < n; i++ {
			for j := 0; j < n; j++ {
				if i == j {
					continue
				}
				col1, col2 := p.Columns[i], p.Columns[j]
				
				series1, _ := df.Column(col1)
				series2, _ := df.Column(col2)
				
				interaction := make([]any, series1.Len())
				for idx := 0; idx < series1.Len(); idx++ {
					v1, ok1 := series1.Get(idx)
					v2, ok2 := series2.Get(idx)
					
					if !ok1 || !ok2 {
						interaction[idx] = nil
						continue
					}
					
					f1 := toFloat64Creator(v1)
					f2 := toFloat64Creator(v2)
					interaction[idx] = f1 * f1 * f2
				}
				
				newColName := fmt.Sprintf("%s^2*%s", col1, col2)
				result = result.WithColumn(newColName, seriesPkg.New(newColName, interaction, core.DtypeFloat64))
			}
		}
	}
	
	return result
}

// FitTransform fits the transformer and transforms the data in one step.
func (p *PolynomialFeatures) FitTransform(df *dataframe.DataFrame, target ...string) (*dataframe.DataFrame, error) {
	if err := p.Fit(df, target...); err != nil {
		return nil, err
	}
	return p.Transform(df)
}

// IsFitted returns true if the transformer has been fitted.
func (p *PolynomialFeatures) IsFitted() bool {
	return p.fitted
}

func getNumericColumns(df *dataframe.DataFrame) []string {
	numericCols := make([]string, 0)
	cols := df.Columns()
	
	for _, col := range cols {
		colSeries, err := df.Column(col)
		if err != nil {
			continue
		}
		
		// Check if column is numeric
		if isNumericSeries(colSeries) {
			numericCols = append(numericCols, col)
		}
	}
	
	return numericCols
}

func isNumericSeries(series interface{ Len() int; Get(int) (any, bool) }) bool {
	for i := 0; i < series.Len() && i < 10; i++ {
		val, ok := series.Get(i)
		if !ok {
			continue
		}
		
		switch val.(type) {
		case float64, float32, int, int64, int32, int16, int8:
			return true
		default:
			return false
		}
	}
	return false
}

func toFloat64Creator(val any) float64 {
	switch v := val.(type) {
	case float64:
		return v
	case float32:
		return float64(v)
	case int:
		return float64(v)
	case int64:
		return float64(v)
	case int32:
		return float64(v)
	default:
		return 0
	}
}

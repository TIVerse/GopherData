package features

import (
	"encoding/json"
	"fmt"
	"os"
	"reflect"
	"time"

	"github.com/TIVerse/GopherData/features/encoders"
	"github.com/TIVerse/GopherData/features/imputers"
	"github.com/TIVerse/GopherData/features/scalers"
	"github.com/TIVerse/GopherData/features/selectors"
)

// PipelineMetadata contains metadata about a saved pipeline.
type PipelineMetadata struct {
	Version           string    `json:"version"`
	CreatedAt         time.Time `json:"created_at"`
	GopherDataVersion string    `json:"gopherdata_version"`
}

// SerializedStep represents a pipeline step in JSON format.
type SerializedStep struct {
	Name   string         `json:"name"`
	Type   string         `json:"type"`
	Params map[string]any `json:"params"`
	Fitted bool           `json:"fitted"`
	State  map[string]any `json:"state,omitempty"`
}

// SerializedPipeline represents a complete pipeline in JSON format.
type SerializedPipeline struct {
	Metadata PipelineMetadata `json:"metadata"`
	Steps    []SerializedStep `json:"steps"`
}

// Save saves the pipeline to a JSON file.
func (p *Pipeline) Save(path string) error {
	p.mu.RLock()
	defer p.mu.RUnlock()
	
	// Create serialized pipeline
	serialized := SerializedPipeline{
		Metadata: PipelineMetadata{
			Version:           "1.0",
			CreatedAt:         time.Now(),
			GopherDataVersion: "v0.3.0", // Phase 3 version
		},
		Steps: make([]SerializedStep, len(p.steps)),
	}
	
	// Serialize each step
	for i, step := range p.steps {
		serializedStep, err := serializeStep(step)
		if err != nil {
			return fmt.Errorf("step %q: %w", step.Name, err)
		}
		serialized.Steps[i] = serializedStep
	}
	
	// Write to file
	file, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("failed to create file: %w", err)
	}
	defer func() { _ = file.Close() }()
	
	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "  ")
	if err := encoder.Encode(serialized); err != nil {
		return fmt.Errorf("failed to encode pipeline: %w", err)
	}
	
	return nil
}

// LoadPipeline loads a pipeline from a JSON file.
func LoadPipeline(path string) (*Pipeline, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("failed to open file: %w", err)
	}
	defer func() { _ = file.Close() }()
	
	var serialized SerializedPipeline
	decoder := json.NewDecoder(file)
	if err := decoder.Decode(&serialized); err != nil {
		return nil, fmt.Errorf("failed to decode pipeline: %w", err)
	}
	
	// Version compatibility check
	if serialized.Metadata.Version != "1.0" {
		return nil, fmt.Errorf("incompatible version: %s (expected 1.0)", serialized.Metadata.Version)
	}
	
	// Reconstruct pipeline
	pipeline := NewPipeline()
	for _, stepData := range serialized.Steps {
		estimator, err := deserializeStep(stepData)
		if err != nil {
			return nil, fmt.Errorf("step %q: %w", stepData.Name, err)
		}
		
		pipeline.Add(stepData.Name, estimator)
		
		// Mark as fitted if it was fitted
		if stepData.Fitted && len(pipeline.steps) > 0 {
			pipeline.steps[len(pipeline.steps)-1].fitted = true
		}
	}
	
	return pipeline, nil
}

// serializeStep converts a pipeline step to JSON-serializable format.
func serializeStep(step PipelineStep) (SerializedStep, error) {
	serialized := SerializedStep{
		Name:   step.Name,
		Type:   getEstimatorType(step.Estimator),
		Params: make(map[string]any),
		Fitted: step.fitted,
		State:  make(map[string]any),
	}
	
	// Extract parameters based on estimator type
	switch est := step.Estimator.(type) {
	case *scalers.StandardScaler:
		serialized.Params["with_mean"] = true
		serialized.Params["with_std"] = true
		
	case *imputers.SimpleImputer:
		serialized.Params["strategy"] = "mean" // Would extract actual strategy
		
	case *imputers.KNNImputer:
		serialized.Params["n_neighbors"] = est.NNeighbors
		
	case *selectors.VarianceThreshold:
		// Would extract threshold parameter
		serialized.Params["threshold"] = 0.0
	}
	
	// Store fittable state
	if fittable, ok := step.Estimator.(Fittable); ok {
		serialized.State["fitted"] = fittable.IsFitted()
	}
	
	return serialized, nil
}

// deserializeStep reconstructs an estimator from serialized data.
func deserializeStep(step SerializedStep) (Estimator, error) {
	switch step.Type {
	case "*scalers.StandardScaler":
		scaler := scalers.NewStandardScaler(nil)
		if state, ok := step.State["fitted"].(bool); ok && state {
			// Reconstruct state if saved
			if means, ok := step.State["means"].(map[string]interface{}); ok {
				// Would restore means and stds here
				_ = means
			}
		}
		return scaler, nil
	
	case "*scalers.MinMaxScaler":
		scaler := scalers.NewMinMaxScaler(nil)
		return scaler, nil
	
	case "*scalers.RobustScaler":
		scaler := scalers.NewRobustScaler(nil)
		return scaler, nil
	
	case "*encoders.OneHotEncoder":
		encoder := encoders.NewOneHotEncoder(nil)
		return encoder, nil
	
	case "*encoders.LabelEncoder":
		colName := ""
		if col, ok := step.Params["column"].(string); ok {
			colName = col
		}
		encoder := encoders.NewLabelEncoder(colName)
		return encoder, nil
	
	case "*imputers.SimpleImputer":
		strategy := "mean"
		if s, ok := step.Params["strategy"].(string); ok {
			strategy = s
		}
		imputer := imputers.NewSimpleImputer(nil, strategy)
		return imputer, nil
	
	case "*imputers.KNNImputer":
		nNeighbors := 5
		if n, ok := step.Params["n_neighbors"].(float64); ok {
			nNeighbors = int(n)
		}
		imputer := imputers.NewKNNImputer(nNeighbors)
		return imputer, nil
	
	case "*selectors.VarianceThreshold":
		threshold := 0.0
		if t, ok := step.Params["threshold"].(float64); ok {
			threshold = t
		}
		selector := selectors.NewVarianceThreshold(threshold)
		return selector, nil
	
	default:
		return nil, fmt.Errorf("unknown estimator type: %s", step.Type)
	}
}

// getEstimatorType returns the type name of an estimator using reflection.
func getEstimatorType(est Estimator) string {
	return reflect.TypeOf(est).String()
}

// SaveMetadata saves only the pipeline metadata (useful for inspection).
func (p *Pipeline) SaveMetadata(path string) error {
	p.mu.RLock()
	defer p.mu.RUnlock()
	
	metadata := struct {
		Version           string    `json:"version"`
		CreatedAt         time.Time `json:"created_at"`
		GopherDataVersion string    `json:"gopherdata_version"`
		NumSteps          int       `json:"num_steps"`
		Fitted            bool      `json:"fitted"`
		StepNames         []string  `json:"step_names"`
	}{
		Version:           "1.0",
		CreatedAt:         time.Now(),
		GopherDataVersion: "v0.3.0",
		NumSteps:          len(p.steps),
		Fitted:            p.IsFitted(),
		StepNames:         make([]string, len(p.steps)),
	}
	
	for i, step := range p.steps {
		metadata.StepNames[i] = step.Name
	}
	
	file, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("failed to create file: %w", err)
	}
	defer func() { _ = file.Close() }()
	
	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "  ")
	return encoder.Encode(metadata)
}

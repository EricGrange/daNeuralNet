{**********************************************************************}
{                                                                      }
{    "The contents of this file are subject to the Mozilla Public      }
{    License Version 2.0 (the "License"); you may not use this         }
{    file except in compliance with the License. You may obtain        }
{    a copy of the License at http://www.mozilla.org/MPL/              }
{                                                                      }
{    Software distributed under the License is distributed on an       }
{    "AS IS" basis, WITHOUT WARRANTY OF ANY KIND, either express       }
{    or implied. See the License for the specific language             }
{    governing rights and limitations under the License.               }
{                                                                      }
{    Current maintainer: Eric Grange                                   }
{    https://delphitools.info                                          }
{                                                                      }
{**********************************************************************}
unit daNeuralNet.Dense;

{$i daNN.inc}

interface

uses daNeuralNet, daNeuralNet.Math;

type
   TdaNNDenseLayer = class;

   TdaNNDenseLayer = class (TdaNeuralNetLayer)
      private
         FActivationFn : IdaNNActivationFunction;
         FBiases : ISingleArray;
         FWeights : ISingleMatrix;
         FWeightsPtr : array of PSingleArray;

         FDeltas : ISingleArray;
         FInputErrors : ISingleArray;
         FChanges : array of ISingleArray;
         FLearningRate : Single;
         FMomentum : Single;

      protected
         procedure RandomizeWeights; override;

         procedure Build(options : TdaNeuralNetBuildOptions); override;

         procedure Compute; override;

         procedure AdjustWeights; override;

         procedure AdjustWeightsWithMomentum;
         procedure AdjustWeightsDirect;

      public
         constructor Create(aSize : Integer; const anActivationFn : IdaNNActivationFunction;
                            aLearningRate, aMomentum : Single);

         procedure ExportWeights(const writer : IdaNNWriter); override;
         procedure CalculateDeltas(const outputErrors : ISingleArray); override;

         property ActivationFn : IdaNNActivationFunction read FActivationFn write FActivationFn;
         property LearningRate : Single read FLearningRate write FLearningRate;
         property Momentum : Single read FMomentum write FMomentum;
   end;

   TdaNeuralNetLayerClass = class of TdaNeuralNetLayer;

// ------------------------------------------------------------------
// ------------------------------------------------------------------
// ------------------------------------------------------------------
implementation
// ------------------------------------------------------------------
// ------------------------------------------------------------------
// ------------------------------------------------------------------

// ------------------
// ------------------ TdaNNDenseLayer ------------------
// ------------------

// Create
//
constructor TdaNNDenseLayer.Create(aSize : Integer; const anActivationFn : IdaNNActivationFunction;
                                 aLearningRate, aMomentum : Single);
begin
   inherited Create(aSize);

   FActivationFn := anActivationFn;
   FLearningRate := aLearningrate;
   FMomentum := aMomentum;
end;

// ExportWeights
//
procedure TdaNNDenseLayer.ExportWeights(const writer : IdaNNWriter);
begin
   writer.BeginSet('Weights');
      writer.WriteMatrix(FWeights);
   writer.EndSet;
   writer.BeginSet('Biases');
      writer.WriteArray(FBiases);
   writer.EndSet;
end;

// Build
//
procedure TdaNNDenseLayer.Build(options : TdaNeuralNetBuildOptions);
begin
   Assert(Previous <> nil, ClassName + ' requires a previous layer');

   FBiases := NewSingleArray(Size);
   FWeights := NewSingleMatrix(Previous.Size, Size, [ moPacked ]);
   SetLength(FWeightsPtr, Size);
   for var i := 0 to Size-1 do begin
      FWeightsPtr[i] := FWeights.RowPtr[i];
   end;

   if nnboForTraining in options then begin
      FDeltas := NewSingleArray(Size);
      FInputErrors := NewSingleArray(Previous.Size);

      if Momentum <> 0 then begin
         SetLength(FChanges, Size);
         for var i := 0 to Size-1 do
            FChanges[i] := NewSingleArray(Previous.Size);
      end;
   end;
end;

// RandomizeWeights
//
procedure TdaNNDenseLayer.RandomizeWeights;
begin
   // He initialization for uniform distribution with a bias to cover edge case of very low sizes
   var r := Sqrt(6 / (Previous.Size + Size + 3));
   for var j := 0 to Size-1 do begin
      FBiases[j] := 0;
      for var i := 0 to Previous.Size-1 do
         FWeightsPtr[j][i] := (Random - 0.5 ) * r;
   end;
end;

// Compute
//
procedure TdaNNDenseLayer.Compute;
begin
   FWeights.MultiplyVector(Previous.Outputs, Self.Outputs);

   var outputsPtr := Outputs.Ptr;
   var biasesPtr := FBiases.Ptr;
   for var node := 0 to Size-1 do begin
      outputsPtr[node] := ActivationFn.Activation(
           biasesPtr[node] + outputsPtr[node]
         );
   end;
end;

// CalculateDeltas
//
procedure TdaNNDenseLayer.CalculateDeltas(const outputErrors : ISingleArray);
begin
   ActivationFn.CalculateDeltas(outputErrors, Outputs, FDeltas);

   if (Previous = nil) or Previous.IsInputLayer then Exit;

   FWeights.TransposeMultiplyVector(FDeltas, FInputErrors);

   Previous.CalculateDeltas(FInputErrors);
end;

// AdjustWeights
//
procedure TdaNNDenseLayer.AdjustWeights;
begin
   if Momentum > 0 then
      AdjustWeightsWithMomentum
   else AdjustWeightsDirect;
end;

// AdjustWeightsDirect
//
procedure TdaNNDenseLayer.AdjustWeightsDirect;
begin
   var rate := LearningRate * Model.LearningRateScale;
   var incomingPtr := Previous.Outputs.Ptr;
   var biases := FBiases.Ptr;
   for var node := 0 to Size-1 do begin
      var delta : Single := rate * FDeltas[node];
      if delta <> 0 then begin
         FWeights.AddScaledVectorToRow(delta, Previous.Outputs, node);
         biases[node] := biases[node] + delta;
      end;
   end;
end;

// AdjustWeightsWithMomentum
//
procedure TdaNNDenseLayer.AdjustWeightsWithMomentum;
begin
   var rate := LearningRate * Model.LearningRateScale;
   var incoming := Previous.Outputs;
   for var node := 0 to Size-1 do begin
      var delta : Single := rate * FDeltas[node];
      var weights := FWeightsPtr[node];
      var changes := FChanges[node];
      for var k := 0 to incoming.High do begin
         var change := delta * incoming[k] + Momentum * changes[k];
         changes[k] := change;
         weights[k] := weights[k] + change;
      end;
      FBiases[node] := FBiases[node] + delta;
   end;
end;

end.

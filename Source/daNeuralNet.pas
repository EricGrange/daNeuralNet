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
unit daNeuralNet;

interface

{$i daNN.inc}

uses SysUtils;

type

   TdaNNBooleanArray = array of Boolean;
   TdaNNSingleArray = array of Single;
   TdaNNSingleStaticArray = array [0..(MaxInt shr 2)-1] of Single;
   PdaNNSingle = ^TdaNNSingleStaticArray;

   TdaNNDataset = array of TdaNNSingleArray;

   IdaNNActivationFunction = interface
      ['{7A572866-F8EA-4887-B358-B63BBF0F2B80}']
      function Activation(v : Single) : Single;
      function Derivation(v : Single) : Single;
      procedure CalculateDeltas(const outputErrors, outputs : TdaNNSingleArray; var deltas : TdaNNSingleArray);
   end;

   IdaNNWriter = interface
      ['{78F9F5DF-3766-4A4C-AAE8-6E81D7BC149B}']
      procedure BeginSet(const name : String);
      procedure EndSet;
      procedure WriteArray(const data : TdaNNSingleArray);
   end;

   TdaNeuralNet = class;

   TdaNeuralNetBuildOption = ( nnboForTraining );
   TdaNeuralNetBuildOptions = set of TdaNeuralNetBuildOption;

   TdaNeuralNetLayer = class abstract
      private
         FPrevious :  TdaNeuralNetLayer;
         FNext :  TdaNeuralNetLayer;
         FModel : TdaNeuralNet;

         FOutputs : TdaNNSingleArray;
         FSize : Integer;

      protected
         FIsInputLayer : Boolean;

         procedure Build(options : TdaNeuralNetBuildOptions); virtual; abstract;
         procedure RandomizeWeights; virtual; abstract;
         procedure ApplyInput(const data : TdaNNSingleArray); virtual;
         procedure Compute; virtual; abstract;
         procedure AdjustWeights; virtual; abstract;

      public
         constructor Create(aSize : Integer);

         property Model : TdaNeuralNet read FModel write FModel;
         property Previous : TdaNeuralNetLayer read FPrevious;
         property Next : TdaNeuralNetLayer read FNext;
         property IsInputLayer : Boolean read FIsInputLayer;

         property Size : Integer read FSize;
         property Outputs : TdaNNSingleArray read FOutputs;

         procedure ExportWeights(const writer : IdaNNWriter); virtual; abstract;

         procedure CalculateDeltas(const outputErrors : TdaNNSingleArray); virtual; abstract;
   end;

   TdaNNInputLayer = class (TdaNeuralNetLayer)
      protected
         procedure Build(options : TdaNeuralNetBuildOptions); override;
         procedure RandomizeWeights; override;
         procedure ApplyInput(const data : TdaNNSingleArray); override;
         procedure Compute; override;
         procedure AdjustWeights; override;

      public
         constructor Create(aSize : Integer);

         procedure ExportWeights(const writer : IdaNNWriter); override;
         procedure CalculateDeltas(const outputErrors : TdaNNSingleArray); override;
   end;

   TdaNeuralNetState = ( nnsBuilt );
   TdaNeuralNetStates = set of TdaNeuralNetState;

   TdaNeuralNet = class
      private
         FLayers : array of TdaNeuralNetLayer;
         FInputLayer : TdaNeuralNetLayer;
         FOutputLayer : TdaNeuralNetLayer;

         FStates : TdaNeuralNetStates;

         FLearningRateScale : Single;

      protected
         procedure AdjustWeights; virtual;

      public
         constructor Create;

         procedure AddLayer(layer : TdaNeuralNetLayer);
         procedure Build(options : TdaNeuralNetBuildOptions); virtual;

         procedure RandomizeWeights; virtual;

         function Run(const data : TdaNNSingleArray) : TdaNNSingleArray; virtual;

         function Train(const input, target : TdaNNSingleArray) : Double; virtual;
         procedure TrainSet(const inputSet, targetSet : TdaNNDataset); virtual;

         property States : TdaNeuralNetStates read FStates;
         property LearningRateScale : Single read FLearningRateScale write FLearningRateScale;

         property InputLayer : TdaNeuralNetLayer read FInputLayer;
         property OutputLayer : TdaNeuralNetLayer read FOutputLayer;
   end;

type
   daNNActivation = class sealed

      class function Sigmoid : IdaNNActivationFunction; static;
      class function ReLu(leakyAlpha : Single = 0.01) : IdaNNActivationFunction; static;
      class function SoftPlus : IdaNNActivationFunction; static;
      class function TanH : IdaNNActivationFunction; static;

   end;

// ------------------------------------------------------------------
// ------------------------------------------------------------------
// ------------------------------------------------------------------
implementation
// ------------------------------------------------------------------
// ------------------------------------------------------------------
// ------------------------------------------------------------------

uses daNeuralNet.Activation, daNeuralNet.Math;

// MSE
//
function MSE(const errors : TdaNNSingleArray) : Double;
begin
   Result := 0;
   for var i := 0 to High(errors) do
      Result := Result + Sqr(errors[i]);
   Result := Result / Length(errors);
end;

// ------------------
// ------------------ daNNActivation ------------------
// ------------------

// Sigmoid
//
var vNNActivationSigmoid : IdaNNActivationFunction;
class function daNNActivation.Sigmoid : IdaNNActivationFunction;
begin
   if vNNActivationSigmoid = nil then
      vNNActivationSigmoid := TdaNNActivationSigmoid.Create;
   Result := vNNActivationSigmoid;
end;

// ReLu
//
class function daNNActivation.ReLu(leakyAlpha : Single = 0.01) : IdaNNActivationFunction;
begin
   Result := TdaNNActivationReLu.Create(leakyAlpha);
end;

// SoftPlus
//
var vNNActivationSoftPlus : IdaNNActivationFunction;
class function daNNActivation.SoftPlus : IdaNNActivationFunction;
begin
   if vNNActivationSoftPlus = nil then
      vNNActivationSoftPlus := TdaNNActivationSoftPlus.Create;
   Result := vNNActivationSoftPlus;
end;

// TanH
//
var vNNActivationTanH : IdaNNActivationFunction;
class function daNNActivation.TanH : IdaNNActivationFunction;
begin
   if vNNActivationTanH = nil then
      vNNActivationTanH := TdaNNActivationSoftPlus.Create;
   Result := vNNActivationTanH;
end;

// ------------------
// ------------------ TdaNeuralNetLayer ------------------
// ------------------

// Create
//
constructor TdaNeuralNetLayer.Create(aSize : Integer);
begin
   inherited Create;
   FSize := aSize;
   SetLength(FOutputs, Size);
end;

// ApplyInput
//
procedure TdaNeuralNetLayer.ApplyInput(const data : TdaNNSingleArray);
begin
   Assert(False, ClassName + ' does not support applying input');
end;

// ------------------
// ------------------ TdaNeuralNet ------------------
// ------------------

// Create
//
constructor TdaNeuralNet.Create;
begin
   inherited;
   FLearningRateScale := 1;
end;

// AddLayer
//
procedure TdaNeuralNet.AddLayer(layer : TdaNeuralNetLayer);
begin
   Assert(not (nnsBuilt in FStates), 'Cannot add layers to a built model');
   Assert(layer.Model = nil, 'Layer already in a model');

   layer.FModel := Self;
   layer.FPrevious := FOutputLayer;
   if FInputLayer = nil then
      FInputLayer := layer
   else layer.FPrevious.FNext := layer;
   FOutputLayer := layer;

   var n := Length(FLayers);
   SetLength(FLayers, n+1);
   FLayers[n] := layer;
end;

// Build
//
procedure TdaNeuralNet.Build(options : TdaNeuralNetBuildOptions);
begin
   Assert(not (nnsBuilt in FStates), 'Model already built');
   for var layer in FLayers do
      layer.Build(options);
   Include(FStates, nnsBuilt);
end;

// RandomizeWeights
//
procedure TdaNeuralNet.RandomizeWeights;
begin
   Assert((nnsBuilt in FStates), 'Model must be built before weights can be randomized');
   for var layer in FLayers do
      layer.RandomizeWeights;
end;

// Run
//
function TdaNeuralNet.Run(const data : TdaNNSingleArray) : TdaNNSingleArray;
begin
   var layer := InputLayer;
   layer.ApplyInput(data);
   while layer <> nil do begin
      layer.Compute;
      layer := layer.Next;
   end;
   Result := OutputLayer.Outputs;
end;

// Train
//
function TdaNeuralNet.Train(const input, target : TdaNNSingleArray) : Double;
begin
   Assert(Length(input) = InputLayer.Size, 'Input size does not match input layer size');
   Assert(Length(target) = OutputLayer.Size, 'Output size does not match output layer size');

   Run(input);

   var deltas := daNNSubtract(target, OutputLayer.Outputs);
   Result := MSE(deltas);

   OutputLayer.CalculateDeltas(deltas);

   AdjustWeights;
end;

// TrainSet
//
procedure TdaNeuralNet.TrainSet(const inputSet, targetSet : TdaNNDataset);
begin
   Assert(Length(inputSet) = Length(targetSet), 'Mismatched input and target sets');
   for var i := 0 to High(inputSet) do
      Train(inputSet[i], targetSet[i]);
end;

// AdjustWeights
//
procedure TdaNeuralNet.AdjustWeights;
begin
   var layer := InputLayer;
   while layer <> nil do begin
      layer.AdjustWeights;
      layer := layer.Next;
   end;
end;

// ------------------
// ------------------ TdaNNInputLayer ------------------
// ------------------

// Create
//
constructor TdaNNInputLayer.Create(aSize : Integer);
begin
   inherited;
   FIsInputLayer := True;
end;

// ExportWeights
//
procedure TdaNNInputLayer.ExportWeights(const writer : IdaNNWriter);
begin
   // nothing
end;

// Build
//
procedure TdaNNInputLayer.Build;
begin
   // nothing
end;

// RandomizeWeights
//
procedure TdaNNInputLayer.RandomizeWeights;
begin
   // nothing
end;

// ApplyInput
//
procedure TdaNNInputLayer.ApplyInput(const data : TdaNNSingleArray);
begin
   Assert(Length(data) = Size);
   System.Move(data[0], Outputs[0], Size*SizeOf(Single));
end;

// Compute
//
procedure TdaNNInputLayer.Compute;
begin
   // nothing
end;

// CalculateDeltas
//
procedure TdaNNInputLayer.CalculateDeltas(const outputErrors : TdaNNSingleArray);
begin
   // nothing
end;

// AdjustWeights
//
procedure TdaNNInputLayer.AdjustWeights;
begin
   // nothing
end;

end.
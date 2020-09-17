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
unit daNeuralNet.Activation;

{$i daNN.inc}

interface

uses System.Math, daNeuralNet, daNeuralNet.Math;

type

   TdaNNActivationSigmoid = class (TInterfacedObject, IdaNNActivationFunction)
      function Activation(v : Single) : Single;
      function Derivation(v : Single) : Single;
      procedure CalculateDeltas(const outputErrors, outputs : ISingleArray; var deltas : ISingleArray);
   end;

   TdaNNActivationReLu = class (TInterfacedObject, IdaNNActivationFunction)
      FAlpha : Single;
      constructor Create(leakyAlpha : Single);
      function Activation(v : Single) : Single;
      function Derivation(v : Single) : Single;
      procedure CalculateDeltas(const outputErrors, outputs : ISingleArray; var deltas : ISingleArray);
   end;

   TdaNNActivationSoftPlus = class (TInterfacedObject, IdaNNActivationFunction)
      function Activation(v : Single) : Single;
      function Derivation(v : Single) : Single;
      procedure CalculateDeltas(const outputErrors, outputs : ISingleArray; var deltas : ISingleArray);
   end;

// ------------------------------------------------------------------
// ------------------------------------------------------------------
// ------------------------------------------------------------------
implementation
// ------------------------------------------------------------------
// ------------------------------------------------------------------
// ------------------------------------------------------------------

// ------------------
// ------------------ TdaNNActivationSigmoid ------------------
// ------------------

// Activation
//
function TdaNNActivationSigmoid.Activation(v : Single) : Single;
begin
   Result := 1 / (1 + Exp(-v));
end;

// Derivation
//
function TdaNNActivationSigmoid.Derivation(v : Single) : Single;
begin
   Result := v * (1-v);
end;

// CalculateDeltas
//
procedure TdaNNActivationSigmoid.CalculateDeltas(const outputErrors, outputs : ISingleArray; var deltas : ISingleArray);
begin
   var pOutputs := outputs.Ptr;
   var pDeltas := deltas.Ptr;
   var pOutputErrors := outputErrors.Ptr;
   for var i := 0 to deltas.High do begin
      var o := pOutputs[i];
      pDeltas[i] := pOutputErrors[i] * o * (1 - o);
   end;
end;

// ------------------
// ------------------ TdaNNActivationReLu ------------------
// ------------------

// Create
//
constructor TdaNNActivationReLu.Create(leakyAlpha : Single);
begin
   inherited Create;
   FAlpha := leakyAlpha;
end;

// Activation
//
function TdaNNActivationReLu.Activation(v : Single) : Single;
begin
   if v > 0 then
      Result := v
   else Result := FAlpha*v;
end;

// Derivation
//
function TdaNNActivationReLu.Derivation(v : Single) : Single;
begin
   if v > 0 then
      Result := 1
   else Result := FAlpha;
end;

// CalculateDeltas
//
procedure TdaNNActivationReLu.CalculateDeltas(const outputErrors, outputs : ISingleArray; var deltas : ISingleArray);
begin
   var pOutputs := outputs.Ptr;
   var pDeltas := deltas.Ptr;
   var pOutputErrors := outputErrors.Ptr;
   for var i := 0 to deltas.High do begin
      if pOutputs[i] > 0 then
         pDeltas[i] := pOutputErrors[i]
      else pDeltas[i] := pOutputErrors[i] * FAlpha;
   end;
end;

// ------------------
// ------------------ TdaNNActivationSoftPlus ------------------
// ------------------

// Activation
//
function TdaNNActivationSoftPlus.Activation(v : Single) : Single;
begin
   Result := Ln(1 + Exp(v));
end;

// Derivation
//
function TdaNNActivationSoftPlus.Derivation(v : Single) : Single;
begin
   Result := 1 / (1 + Exp(-v));
end;

// CalculateDeltas
//
procedure TdaNNActivationSoftPlus.CalculateDeltas(const outputErrors, outputs : ISingleArray; var deltas : ISingleArray);
begin
   var pOutputs := outputs.Ptr;
   var pDeltas := deltas.Ptr;
   var pOutputErrors := outputErrors.Ptr;
   for var i := 0 to deltas.High do begin
      pDeltas[i] := pOutputErrors[i] / (1 + Exp(-pOutputs[i]));
   end;
end;

end.

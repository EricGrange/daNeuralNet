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
unit daNeuralNet.Pooling;

{$i daNN.inc}

interface

uses daNeuralNet;

type
   TdaNNPoolingCombiner = ( pcMax, pcAverage );

   TdaNNPoolingLayer_2by2 = class (TNeuralNetLayer)
      private
         FCombiner : TdaNNPoolingCombiner;
         FSX, FSY : Integer;

      protected
         procedure Build(options : TNeuralNetBuildOptions); override;

         procedure Compute; override;
         procedure ComputeMax;
         procedure ComputeAverage;

         procedure RandomizeWeights; override;
         procedure AdjustWeights; override;

      public
         constructor Create(sx, sy : Integer; aCombiner : TdaNNPoolingCombiner);

         procedure CalculateDeltas(const outputErrors : TdaNNSingleArray); override;

         property Combiner : TdaNNPoolingCombiner read FCombiner write FCombiner;
   end;

// ------------------------------------------------------------------
// ------------------------------------------------------------------
// ------------------------------------------------------------------
implementation
// ------------------------------------------------------------------
// ------------------------------------------------------------------
// ------------------------------------------------------------------

// ------------------
// ------------------ TdaNNPoolingLayer_2by2 ------------------
// ------------------

// Create
//
constructor TdaNNPoolingLayer_2by2.Create(sx, sy : Integer; aCombiner : TdaNNPoolingCombiner);
begin
   Assert((sx and 1) = 0, ClassName + ' requires sx to be multiple of 2');
   Assert((sy and 1) = 0, ClassName + ' requires sy to be multiple of 2');
   inherited Create(sx*sy div 4);
   FSX := sx;
   FSY := sy;
   FCombiner := aCombiner;
end;

// Build
//
procedure TdaNNPoolingLayer_2by2.Build(options : TNeuralNetBuildOptions);
begin
   Assert(Previous <> nil, ClassName + ' requires a previous layer');
   Assert(Previous.Size = FSX*FSY, Classname + ' mismatched topology with previous layer');
end;

// Compute
//
procedure TdaNNPoolingLayer_2by2.Compute;
begin
   case FCombiner of
      pcMax : ComputeMax;
      pcAverage : ComputeAverage;
   else
      Assert(False);
   end;
end;

// ComputeMax
//
procedure TdaNNPoolingLayer_2by2.ComputeMax;
begin
   var pInput := PNNSingle(Previous.Outputs);
   var pInputNext := PNNSingle(@pInput[FSX]);
   var pOutput := PNNSingle(Outputs);
   var pOutputSize := @pOutput[Size];
   var x2 := (FSX shr 1) - 1;
   while NativeUInt(pOutput) < NativeUInt(pOutputSize) do begin
      for var x := 0 to x2 do begin
         var m, m2 : Single;
         if pInput[0] > pInput[1] then
            m := pInput[0]
         else m := pInput[1];
         if pInputNext[0] > pInputNext[1] then
            m2 := pInputNext[0]
         else m2 := pInputNext[1];
         if m2 > m then
            pOutput[0] := m2
         else pOutput[0] := m;
         pOutput := @pOutput[1];
         pInput := @pInput[2];
         pInputNext := @pInputNext[2]
      end;
      pInput := @pInput[FSX];
      pInputNext := @pInputNext[FSX];
   end;
end;

// ComputeAverage
//
procedure TdaNNPoolingLayer_2by2.ComputeAverage;
begin
   var pInput := PNNSingle(Previous.Outputs);
   var pInputNext := PNNSingle(@pInput[FSX]);
   var pOutput := PNNSingle(Outputs);
   var pOutputSize := @pOutput[Size];
   var x2 := (FSX shr 1) - 1;
   while NativeUInt(pOutput) < NativeUInt(pOutputSize) do begin
      for var x := 0 to x2 do begin
         pOutput[0] := 0.25 * (  pInput[0] + pInput[1]
                               + pInputNext[0] + pInputNext[1]);
         pOutput := @pOutput[1];
         pInput := @pInput[2];
         pInputNext := @pInputNext[2]
      end;
      pInput := @pInput[FSX];
      pInputNext := @pInputNext[FSX];
   end;
end;

// RandomizeWeights
//
procedure TdaNNPoolingLayer_2by2.RandomizeWeights;
begin
   /// nothing
end;

// AdjustWeights
//
procedure TdaNNPoolingLayer_2by2.AdjustWeights;
begin
   // nothing
end;

// CalculateDeltas
//
procedure TdaNNPoolingLayer_2by2.CalculateDeltas(const outputErrors : TdaNNSingleArray);
begin
   // TODO
end;

end.

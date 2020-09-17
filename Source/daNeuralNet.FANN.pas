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
//
// This unit requires https://github.com/FMXExpress/Delphi-FANN
//
unit daNeuralNet.FANN;

interface

uses SysUtils, daNeuralNet, delphi_fann;

type

   TdaNeuralNetFANN = class (TdaNeuralNet)
      private
         FNet : TFannNetwork;

      public
         constructor Create(const neuronsPerLayer : array of const);
         destructor Destroy; override;

         procedure RandomizeWeights; override;

         function Run(const data : TdaNNSingleArray) : TdaNNSingleArray; override;
         function Train(const input, target : TdaNNSingleArray) : Double; override;
   end;

implementation

// ------------------
// ------------------ TdaNeuralNetFANN ------------------
// ------------------

// Create
//
constructor TdaNeuralNetFANN.Create(const neuronsPerLayer : array of const);
begin
   inherited Create;
   FNet := TFannNetwork.Create(nil);

   var n := Length(neuronsPerLayer);

   for var i := 0 to n-1 do begin

      Assert(neuronsPerLayer[i].VType = vtInteger);
      var s := neuronsPerLayer[i].VInteger;

      FNet.Layers.Add(IntToStr(s));

   end;

end;

// Destroy
//
destructor TdaNeuralNetFANN.Destroy;
begin
   inherited;
   FNet.Free;
end;

// RandomizeWeights
//
procedure TdaNeuralNetFANN.RandomizeWeights;
begin
   FNet.Build;
end;

// Run
//
function TdaNeuralNetFANN.Run(const data : TdaNNSingleArray) : TdaNNSingleArray;
begin
   FNet.Run(data, Result);
end;

// Train
//
function TdaNeuralNetFANN.Train(const input, target : TdaNNSingleArray) : Double;
begin
   Result := FNet.Train(input, target);
end;

end.

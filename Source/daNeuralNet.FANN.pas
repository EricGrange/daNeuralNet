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

uses SysUtils, NeuralNet.Base, FannNetwork;

type

   TNeuralNetFANN = class (TNeuralNet)
      private
         FNet : TFannNetwork;

      public
         constructor Create(const neuronsPerLayer : array of const);
         destructor Destroy; override;

         procedure RandomizeWeights; override;

         function Run(const data : TNNSingleArray) : TNNSingleArray; override;
         function Train(const input, target : TNNSingleArray) : Double; override;
   end;

implementation

// ------------------
// ------------------ TNeuralNetFANN ------------------
// ------------------

// Create
//
constructor TNeuralNetFANN.Create(const neuronsPerLayer : array of const);
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
destructor TNeuralNetFANN.Destroy;
begin
   inherited;
   FNet.Free;
end;

// RandomizeWeights
//
procedure TNeuralNetFANN.RandomizeWeights;
begin
   FNet.Build;
end;

// Run
//
function TNeuralNetFANN.Run(const data : TNNSingleArray) : TNNSingleArray;
begin
   FNet.Run(data, Result);
end;

// Train
//
function TNeuralNetFANN.Train(const input, target : TNNSingleArray) : Double;
begin
   Result := FNet.Train(input, target);
end;

end.

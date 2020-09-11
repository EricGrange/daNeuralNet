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
// This unit makes use of DWScript dwsJSON
//
unit daNeuralNet.JSON;

{$i daNN.inc}

interface

uses daNeuralNet, dwsJSON;

type

   TdaNNJSONWriter = class (TInterfacedObject, IdaNNWriter)
      private
         FWriter : TdwsJSONWriter;
      public
         constructor Create(aWriter : TdwsJSONWriter);

         procedure BeginSet(const name : String);
         procedure EndSet;
         procedure WriteArray(const data : TdaNNSingleArray);
   end;

implementation

// ------------------
// ------------------ TdaNNJSONWriter ------------------
// ------------------

// Create
//
constructor TdaNNJSONWriter.Create(aWriter : TdwsJSONWriter);
begin
   inherited Create;
   FWriter := aWriter;
end;

// BeginSet
//
procedure TdaNNJSONWriter.BeginSet(const name : String);
begin
   FWriter.WriteName(name).BeginArray;
end;

// EndSet
//
procedure TdaNNJSONWriter.EndSet;
begin
   FWriter.EndArray;
end;

// WriteArray
//
procedure TdaNNJSONWriter.WriteArray(const data : TdaNNSingleArray);
begin
   FWriter.BeginArray;
   for var i := 0 to High(data) do
      FWriter.WriteNumber(data[i]);
   FWriter.EndArray;
end;

end.

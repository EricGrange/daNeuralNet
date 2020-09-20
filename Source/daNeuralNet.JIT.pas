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
unit daNeuralNet.JIT;

{$i daNN.inc}

interface

uses
   Windows, SysUtils, Classes,
   daNeuralNet.Math, dwsJITx86Intrinsics;

type

   IdaNNJIT = interface
      ['{CA1C0ECD-00B1-429D-8BD5-8ACA6736741B}']
      function Ptr : Pointer;
   end;

   TdaNNJIT = class (TInterfacedObject, IdaNNJIT)
      private
         FBuffer : Tx86_Platform_WriteOnlyStream;
         FPtr : Pointer;

      public
         constructor Create;
         destructor Destroy; override;

         procedure Build;
         function Ptr : Pointer;

         property Buffer : Tx86_Platform_WriteOnlyStream read FBuffer;
         function Position : Integer;
   end;


type
   TCompiledDotProduct = function (p1, p2 : PSingleArray) : Single; register;
function CompileDotProduct(nb : Integer) : IdaNNJIT;

implementation

// CompileDotProduct
//
function CompileDotProduct(nb : Integer) : IdaNNJIT;
begin
   var nnJIT := TdaNNJIT.Create;
   Result := nnJIT;
   var jit := nnJIT.Buffer;

   var useSimd := False;
   var majorLoopStep := 0;
   var tailOffset := 0;

   jit._xorps_reg_reg(xmm0, xmm0);

   if nb >= 8 then begin
      majorLoopStep := 8;
      jit._xorps_reg_reg(xmm2, xmm2);
   end;

   if majorLoopStep > 0 then begin

      useSimd := True;

      var majorLoopCount := nb div majorLoopStep;

      if majorLoopCount > 1 then
         jit._mov_reg_dword(gprEAX, majorLoopCount);

      var jumpRef := jit.Position;

      jit._movups_reg_ptr_reg(xmm1, gprRCX, 0);
      jit._mulps_reg_ptr_reg(xmm1, gprRDX, 0);
      jit._addps_reg_reg(xmm0, xmm1);

      jit._movups_reg_ptr_reg(xmm3, gprRCX, 4*SizeOf(Single));
      jit._mulps_reg_ptr_reg(xmm3, gprRDX, 4*SizeOf(Single));
      jit._addps_reg_reg(xmm2, xmm3);

      nb := nb mod majorLoopStep;

      if majorLoopCount > 1 then begin
         jit._add_reg_int32(gprRCX, majorLoopStep*SizeOf(Single));
         jit._add_reg_int32(gprRDX, majorLoopStep*SizeOf(Single));
         jit._dec(gprEAX);
         jit._jump(flagsNZ, jumpRef-jit.Position);
      end else begin
         tailOffset := majorLoopStep*SizeOf(Single);
      end;

      jit._addps_reg_reg(xmm0, xmm2);
   end;

   if nb >= 4 then begin
      useSimd := True;
      jit._movups_reg_ptr_reg(xmm4, gprRCX, tailOffset);
      jit._mulps_reg_ptr_reg(xmm4, gprRDX, tailOffset);
      jit._addps_reg_reg(xmm0, xmm4);
      Inc(tailOffset, 4*SizeOf(Single));
      nb := nb and 3;
   end;

   if useSimd then begin
      jit._movshdup(xmm7, xmm0);
      jit._addps_reg_reg(xmm0, xmm7);
      jit._movhlps(xmm7, xmm0);
      jit._addss(xmm0, xmm7);
   end;

   for var i := 0 to nb-1 do begin
      var xmmReg := TxmmRegister(Ord(xmm1) + (i and 3));
      jit._movss_reg_ptr_reg(xmmReg, gprRCX, tailOffset + i*SizeOf(Single));
      jit._mulss_reg_ptr_reg(xmmReg, gprRDX, tailOffset + i*SizeOf(Single));
      jit._addss(xmm0, xmmReg);
   end;

   nnJIT.Build;
end;

// ------------------
// ------------------ TdaNNJIT ------------------
// ------------------

// Create
//
constructor TdaNNJIT.Create;
begin
   inherited;
   FBuffer := Tx86_Platform_WriteOnlyStream.Create;
end;

// Destroy
//
destructor TdaNNJIT.Destroy;
begin
   inherited;
   if Assigned(FPtr) then
      VirtualFree(FPtr, 0 , MEM_RELEASE);
end;

// Build
//
procedure TdaNNJIT.Build;
begin
   Assert(FPtr = nil, 'Already built');

   FBuffer._ret;
   FBuffer._nop(7);

   var opcodes := FBuffer.ToRawBytes;
   var n := Length(opcodes);
   FPtr := VirtualAlloc(nil, n, MEM_COMMIT, PAGE_READWRITE);
   System.Move(Pointer(opcodes)^, FPtr^, n);
   var oldProtect : Integer;
   VirtualProtect(FPtr, n, PAGE_EXECUTE_READ, @oldProtect);

   FreeAndNil(FBuffer);
end;

// Ptr
//
function TdaNNJIT.Ptr : Pointer;
begin
   Result := FPtr;
end;

// Position
//
function TdaNNJIT.Position : Integer;
begin
   Result := FBuffer.Position;
end;

end.

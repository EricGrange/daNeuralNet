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

   jit._xorps_reg_reg(xmm0, xmm0);

   var useSimd := (nb >= 8);
   var majorLoop := nb div (4*7);

   if majorLoop > 1 then
      jit._mov_reg_dword(gprEAX, majorLoop);

   var jumpRef := jit.Position;

   while useSimd and (nb >= 4) do begin
      jit._movaps_reg_ptr_reg(xmm1, gprRCX, 0);
      jit._mulps_reg_ptr_reg(xmm1, gprRDX, 0);
      jit._addps_reg_reg(xmm0, xmm1);
      var processed := 4;

      while (nb >= processed + 4) and (processed <= 6*4) do begin
         var xmmReg := TxmmRegister((processed shr 2) + 1);
         jit._movaps_reg_ptr_reg(TxmmRegister(xmmReg), gprRCX, processed*SizeOf(Single));
         jit._mulps_reg_ptr_reg(TxmmRegister(xmmReg), gprRDX, processed*SizeOf(Single));
         jit._addps_reg_reg(xmm0, xmmReg);
         Inc(processed, 4);
      end;

      Dec(nb, processed);

      if nb > 0 then begin
         jit._add_reg_int32(gprRCX, processed*SizeOf(Single));
         jit._add_reg_int32(gprRDX, processed*SizeOf(Single));
      end;

      if majorLoop > 1 then
         Break;
   end;
   if majorLoop > 1 then begin
      jit._dec(gprEAX);
      jit._jump(flagsNZ, jumpRef-jit.Position);
      Dec(nb, (majorLoop-1) * 28)
   end;

   if useSimd then begin
      jit._movshdup(xmm1, xmm0);
      jit._addps_reg_reg(xmm0, xmm1);
      jit._movhlps(xmm1, xmm0);
      jit._addss(xmm0, xmm1);
   end;

   if nb > 0 then begin
      for var i := 0 to nb-1 do begin
         jit._movss_reg_ptr_reg(xmm1, gprRCX, i*SizeOf(Single));//
         jit._mulss_reg_ptr_reg(xmm1, gprRDX, i*SizeOf(Single));//
         jit._addss(xmm0, xmm1);
      end;
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

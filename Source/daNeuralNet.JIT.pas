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
      function Size : Integer;
   end;

   TdaNNJIT = class (TInterfacedObject, IdaNNJIT)
      private
         FBuffer : Tx86_Platform_WriteOnlyStream;
         FPtr : Pointer;
         FSize : Integer;

      public
         constructor Create;
         destructor Destroy; override;

         procedure Build;
         function Ptr : Pointer;
         function Size : Integer;

         property Buffer : Tx86_Platform_WriteOnlyStream read FBuffer;
         function Position : Integer;
   end;


type
   TCompiledDotProduct = function (p1, p2 : PSingleArray) : Single; register;
   TCompiledMatVectorMult = procedure (pMat, pVec, pRes : PSingleArray); register;

function CompileMultVectorAVX_enh(nbCol, nbRow : Integer; colAlignedSize : Integer = 0) : IdaNNJIT;
function CompileMultVectorAVX(nbCol, nbRow : Integer; colAlignedSize : Integer = 0) : IdaNNJIT;
function CompileDotProductSSE(nb : Integer) : IdaNNJIT;

implementation

// CompileMultVectorAVX
//
function CompileMultVectorAVX_enh(nbCol, nbRow : Integer; colAlignedSize : Integer = 0) : IdaNNJIT;
begin
   // matrix in RCX
   // vector in RDX
   // result in R8

   var nnJIT := TdaNNJIT.Create;
   Result := nnJIT;
   var jit := nnJIT.Buffer;

   if nbRow > 4 then
      jit._mov_reg_reg(gprR9, gprRDX);

   var colLoopStep := 8;

   for var k := 0 to 1 do begin

      var rowLoopStep := 4;
      if rowLoopStep > nbRow then
         rowLoopStep := nbRow;

      var rowLoopCount := nbRow div rowLoopStep;

      if rowLoopCount > 1 then
         jit._mov_reg_dword(gprR10d, rowLoopCount);
      var rowLoopRef := jit.Position;

      var colRemaining := nbCol mod colLoopStep;
      var colOffset := 0;
      var colCatchup := 0;

      var colLoopCount := nbCol div colLoopStep;

      if colLoopCount > 1 then begin

         jit._vmovups_ptr_reg(ymm14, gprRDX, 0);
         for var i := 0 to rowLoopStep-1 do begin
            jit._vmulps_ptr_reg(TymmRegister(i), ymm14, gprRCX, i*nbCol*SizeOf(Single));
         end;
         jit._add_reg_int32(gprRDX, colLoopStep*SizeOf(Single));
         jit._add_reg_int32(gprRCX, colLoopStep*SizeOf(Single));

         if colLoopCount > 1 then begin
            if colLoopCount > 2 then
               jit._mov_reg_dword(gprEAX, colLoopCount-1);

            var colLoopRef := jit.Position;

            jit._vmovups_ptr_reg(ymm15, gprRDX, 0);

            for var i := 0 to rowLoopStep-1 do begin
               jit._vfmadd231ps_ptr_reg(
                  TymmRegister(i), ymm15, gprRCX, i*nbCol*SizeOf(Single)
               );
            end;

            jit._add_reg_int32(gprRDX, colLoopStep*SizeOf(Single));
            jit._add_reg_int32(gprRCX, colLoopStep*SizeOf(Single));
            if colLoopCount > 2 then begin
               jit._dec(gprEAX);
               jit._jump(flagsNZ, colLoopRef-jit.Position);
            end;
         end;

         for var i := 0 to rowLoopStep-1 do begin
            var highBufReg := TxmmRegister(rowLoopStep + i);
            jit._vextract128_high(highBufReg, TymmRegister(i));
            jit._vaddps(TxmmRegister(i), TxmmRegister(i), highBufReg);
         end;

      end else begin

         colCatchup := colLoopCount*colLoopStep;
         Inc(colRemaining, colCatchup);

         if colRemaining >= 4 then begin
            jit._vmovups_ptr_reg(xmm9, gprRDX, colOffset);
            for var i := 0 to rowLoopStep-1 do
               jit._vmulps_ptr_reg(TxmmRegister(i), xmm9, gprRCX, i*nbCol*SizeOf(Single));
            Inc(colOffset, 4*SizeOf(Single));
            Dec(colRemaining, 4);
         end else begin
            for var i := 0 to rowLoopStep-1 do
               jit._vxorps(TymmRegister(i));
         end;

      end;

      while colRemaining >= 4 do begin
         jit._vmovups_ptr_reg(xmm10, gprRDX, colOffset);
         for var i := 0 to rowLoopStep-1 do begin
            jit._vfmadd231ps_ptr_reg(
               TxmmRegister(i), xmm10, gprRCX, colOffset + i*nbCol*SizeOf(Single)
            );
         end;
         Inc(colOffset, 4*SizeOf(Single));
         Dec(colRemaining, 4);
      end;

      for var i := 0 to rowLoopStep-1 do begin
         var reg := TxmmRegister(i);
         var bufReg := TxmmRegister(rowLoopStep+i);
         jit._vmovshdup(bufReg, reg);
         jit._vaddps(reg, bufReg, reg);
         jit._vmovhlps(bufReg, bufReg, reg);
         jit._vaddss(reg, reg, bufReg);
      end;

      while colRemaining > 0 do begin
         jit._vmovss_reg_ptr_reg(xmm8, gprRDX, colOffset);
         for var i := 0 to rowLoopStep-1 do begin
            jit._vfmadd231ss_ptr_reg(
               TxmmRegister(i), xmm8,
               gprRCX, colOffset + i*nbCol*SizeOf(Single)
            );
         end;
         Inc(colOffset, SizeOf(Single));
         Dec(colRemaining);
      end;

      for var i := 0 to rowLoopStep-1 do
         jit._vmovss_ptr_reg_reg(gprR8, i*SizeOf(Single), TxmmRegister(i));

      if k = 0 then begin
         nbRow := nbRow mod rowLoopStep;

         if (nbRow > 0) or (rowLoopCount > 1) then begin
            jit._add_reg_int32(gprRCX, SizeOf(Single)*(nbCol*(rowLoopStep-1) + nbCol mod colLoopStep + colCatchup));
            jit._mov_reg_reg(gprRDX, gprR9);
            jit._add_reg_int32(gprR8, rowLoopStep*SizeOf(Single));
         end;

         if rowLoopCount > 1 then begin
            jit._dec(gprR10d);
            jit._jump(flagsNZ, rowLoopRef-jit.Position);
         end;

         if nbRow = 0 then break;
      end;
   end;

   jit._vzeroupper;

   nnJIT.Build;
end;

// CompileMultVectorAVX
//
function CompileMultVectorAVX(nbCol, nbRow : Integer; colAlignedSize : Integer = 0) : IdaNNJIT;
begin
   // matrix in RCX
   // vector in RDX
   // result in R8

   var nnJIT := TdaNNJIT.Create;
   Result := nnJIT;
   var jit := nnJIT.Buffer;

   if nbRow > 4 then
      jit._mov_reg_reg(gprR9, gprRDX);

   var colLoopStep := 8;

   for var k := 0 to 1 do begin

      var rowLoopStep := 4;
      if rowLoopStep > nbRow then
         rowLoopStep := nbRow;

      var rowLoopCount := nbRow div rowLoopStep;

      if rowLoopCount > 1 then
         jit._mov_reg_dword(gprR10d, rowLoopCount);
      var rowLoopRef := jit.Position;

      var colRemaining := nbCol mod colLoopStep;
      var colOffset := 0;

      var colLoopCount := nbCol div colLoopStep;

      if colLoopCount > 0 then begin

         if colLoopCount > 1 then
            jit._mov_reg_dword(gprEAX, colLoopCount);

         for var i := 0 to rowLoopStep-1 do begin
            jit._vxorps(TymmRegister(i));
         end;

         var colLoopRef := jit.Position;

         jit._vmovups_ptr_reg(ymm15, gprRDX, 0);

         for var i := 0 to rowLoopStep-1 do begin
            jit._vfmadd231ps_ptr_reg(
               TymmRegister(i), ymm15,
               gprRCX, i*nbCol*SizeOf(Single)
            );
         end;

         jit._add_reg_int32(gprRDX, colLoopStep*SizeOf(Single));
         jit._add_reg_int32(gprRCX, colLoopStep*SizeOf(Single));
         if colLoopCount > 1 then begin
            jit._dec(gprEAX);
            jit._jump(flagsNZ, colLoopRef-jit.Position);
         end;

         for var i := 0 to rowLoopStep-1 do begin
            var highBufReg := TxmmRegister(4 + i);
            jit._vextract128_high(highBufReg, TymmRegister(i));
            jit._vaddps(TxmmRegister(i), TxmmRegister(i), highBufReg);
         end;

      end else begin

         for var i := 0 to rowLoopStep-1 do
            jit._xorps_reg_reg(TxmmRegister(i), TxmmRegister(i));

      end;

      while colRemaining >= 4 do begin
         jit._vmovups_ptr_reg(xmm8, gprRDX, colOffset);
         for var i := 0 to rowLoopStep-1 do begin
            jit._vfmadd231ps_ptr_reg(
               TxmmRegister(i), xmm8,
               gprRCX, colOffset + i*nbCol*SizeOf(Single)
            );
         end;
         Inc(colOffset, 4*SizeOf(Single));
         Dec(colRemaining, 4);
      end;

      for var i := 0 to rowLoopStep-1 do begin
         var reg := TxmmRegister(i);
         var bufReg := TxmmRegister(4+i);
         jit._vmovshdup(bufReg, reg);
         jit._vaddps(reg, bufReg, reg);
         jit._vmovhlps(bufReg, bufReg, reg);
         jit._vaddss(reg, reg, bufReg);
      end;

      while colRemaining > 0 do begin
         jit._vmovss_reg_ptr_reg(xmm8, gprRDX, colOffset);
         for var i := 0 to rowLoopStep-1 do begin
            jit._vfmadd231ss_ptr_reg(
               TxmmRegister(i), xmm8,
               gprRCX, colOffset + i*nbCol*SizeOf(Single)
            );
         end;
         Inc(colOffset, SizeOf(Single));
         Dec(colRemaining);
      end;

      for var i := 0 to rowLoopStep-1 do
         jit._vmovss_ptr_reg_reg(gprR8, i*SizeOf(Single), TxmmRegister(i));

      if k = 0 then begin
         nbRow := nbRow mod rowLoopStep;

         if (nbRow > 0) or (rowLoopCount > 1) then begin
            jit._add_reg_int32(gprRCX, SizeOf(Single)*(nbCol*(rowLoopStep-1) + nbCol mod colLoopStep));
            jit._mov_reg_reg(gprRDX, gprR9);
            jit._add_reg_int32(gprR8, rowLoopStep*SizeOf(Single));
         end;

         if rowLoopCount > 1 then begin
            jit._dec(gprR10d);
            jit._jump(flagsNZ, rowLoopRef-jit.Position);
         end;

         if nbRow = 0 then break;
      end;
   end;

   jit._vzeroupper;

   nnJIT.Build;
end;

// CompileDotProductSSE
//
function CompileDotProductSSE(nb : Integer) : IdaNNJIT;
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

   while nb >= 4 do begin
      useSimd := True;
      jit._movups_reg_ptr_reg(xmm4, gprRCX, tailOffset);
      jit._mulps_reg_ptr_reg(xmm4, gprRDX, tailOffset);
      jit._addps_reg_reg(xmm0, xmm4);
      Inc(tailOffset, 4*SizeOf(Single));
      Dec(nb, 4);
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
   FBuffer._nop(7-(FBuffer.Position and 7)); // nop fill 8 bytes line

   var opcodes := FBuffer.ToRawBytes;
   FSize := Length(opcodes);
   FPtr := VirtualAlloc(nil, FSize, MEM_COMMIT, PAGE_READWRITE);
   System.Move(Pointer(opcodes)^, FPtr^, FSize);
   var oldProtect : Integer;
   VirtualProtect(FPtr, FSize, PAGE_EXECUTE_READ, @oldProtect);

   FreeAndNil(FBuffer);
end;

// Ptr
//
function TdaNNJIT.Ptr : Pointer;
begin
   Result := FPtr;
end;

// Size
//
function TdaNNJIT.Size : Integer;
begin
   Result := FSize;
end;

// Position
//
function TdaNNJIT.Position : Integer;
begin
   Result := FBuffer.Position;
end;

end.

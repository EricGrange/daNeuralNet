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
unit daNeuralNet.Math;

{$i daNN.inc}

interface

uses SysUtils, Classes;

type
   TSingleArray = array [0 .. MaxInt shr 3] of Single;
   PSingleArray = ^TSingleArray;

   ISingleArray = interface
      ['{EB16BFBE-B2F6-4BDB-B7BF-A302DD26EC5E}']
      function Length : Integer;
      function High : Integer;
      function GetItem(index : Integer) : Single;
      procedure SetItem(index : Integer; v : Single);
      property Items[index : Integer] : Single read GetItem write SetItem; default;
      function Ptr : PSingleArray;

      function SumOfSquares : Double;
   end;

   ISingleMatrix = interface
      ['{BDF70136-7F1F-4E3F-B677-35486C48952D}']
      function ColumnCount : Integer;
      function AlignedColumnCount : Integer;
      function RowCount : Integer;
      function Count : Integer;
      function GetItem(col, row : Integer) : Single;
      procedure SetItem(col, row : Integer; v : Single);
      property Items[col, row : Integer] : Single read GetItem write SetItem; default;
      function RowPtr(row : Integer) : PSingleArray;

      procedure MultiplyVector(const vector, result : ISingleArray);
   end;

function NewSingleArray(size : Integer) : ISingleArray;

function NewSingleMatrix(colCount, rowCount : Integer) : ISingleMatrix;

function  daNNDotProduct(p1, p2 : PSingleArray; nb : Integer) : Single;

procedure daNNAddScaledOperand(pTarget, pOperand : PSingleArray; scale : Single; nb : Integer);
function  daNNSubtract(const left, right : ISingleArray) : ISingleArray;

procedure daNNMathSelfTest;

// ------------------------------------------------------------------
// ------------------------------------------------------------------
// ------------------------------------------------------------------
implementation
// ------------------------------------------------------------------
// ------------------------------------------------------------------
// ------------------------------------------------------------------

uses daNeuralNet.JIT;

// daNNDotProduct
//
function daNNDotProduct(p1, p2 : PSingleArray; nb : Integer) : Single;
{$if Defined(WIN64_ASM)}
asm
      pxor  xmm0, xmm0
      cmp   r8d, 3 + 4  // a single loop4 is not beneficial
      jle   @@tail3

      mov   eax, r8d
      shr   eax, 2
      and   r8d, 3

   @@loop4:
      movaps   xmm1, [rcx]
      mulps    xmm1, [rdx]
      addps    xmm0, xmm1

      add   rcx, 16
      add   rdx, 16
      dec   eax
      jnz   @@loop4

//      pshufd   xmm1, xmm0, $E
//      addps    xmm0, xmm1
//      pshufd   xmm1, xmm0, $1
//      addss    xmm0, xmm1
      movshdup xmm1, xmm0
      addps xmm0, xmm1
      movhlps xmm1, xmm0
      addss xmm0, xmm1

   @@tail3:
      test  r8d, r8d
      jz    @@done

   @@loop:
      movss xmm1, [rcx]
      mulss xmm1, [rdx]
      addss xmm0, xmm1
      add   rcx, 4
      add   rdx, 4
      dec   r8d
      jnz   @@loop

   @@done:
end;
{$elseif Defined(WIN32_ASM)}
asm
      pxor  xmm0, xmm0
      cmp   ecx, 3 + 4  // a single loop4 is not beneficial
      jle   @@tail3

      push  ecx
      shr   ecx, 2

   @@loop4:
      movaps   xmm1, [eax]
      movaps   xmm2, [edx]
      mulps    xmm1, xmm2
      addps    xmm0, xmm1

      add   eax, 16
      add   edx, 16
      dec   ecx
      jnz   @@loop4

      pshufd   xmm1, xmm0, $E
      addps    xmm0, xmm1
      pshufd   xmm1, xmm0, $1
      addss    xmm0, xmm1

      pop   ecx
      and   ecx, 3

   @@tail3:
      test  ecx, ecx
      jz    @@done

   @@loop:
      movss xmm1, [eax + ecx * 4 - 4]
      mulss xmm1, [edx + ecx * 4 - 4]
      addss xmm0, xmm1
      dec   ecx
      jnz   @@loop

   @@done:
      push  eax
      movss [esp], xmm0
      fld   dword ptr [esp]
      pop   eax
end;
(*
asm
      fldz
      test  ecx, ecx
      jz    @@done

   @@loop:
      fld   dword ptr [eax + ecx*4 - 4]
      fmul  dword ptr [edx + ecx*4 - 4]
      faddp
      dec ecx
      jnz @@loop

   @@done:
end;*)
{$else}
begin
   Result := 0;
   while nb > 0 do begin
      Dec(nb);
      Result := Result + p1^ * p2^;
      Inc(p1);
      Inc(p2);
   end;
end;
{$endif}

// daNNAddScaledOperand
//
procedure daNNAddScaledOperand(pTarget, pOperand : PSingleArray; scale : Single; nb : Integer);
{$if Defined(WIN64_ASM)}
asm
      cmp   r9d, 3
      jle   @@tail3

      mov   eax, r9d
      shr   eax, 2
      and   r9d, 3

      shufps xmm2, xmm2, 0

   @@loop4:
      movaps   xmm0, [rcx]
      movaps   xmm1, [rdx]
      mulps    xmm1, xmm2
      addps    xmm0, xmm1
      movaps   [rcx], xmm0

      add   rcx, 16
      add   rdx, 16
      dec   eax
      jnz   @@loop4

   @@tail3:
      test  r9d, r9d
      jz    @@done

   @@loop:
      movss xmm0, [rcx]
      movss xmm1, [rdx]
      mulss xmm1, xmm2
      addss xmm0, xmm1
      movss [rcx], xmm0
      add   rcx, 4
      add   rdx, 4
      dec   r9d
      jnz   @@loop

   @@done:
end;
{$elseif Defined(WIN32_ASM)}
asm
      push  edi
      mov   edi, eax

      movss xmm2, scale
      mov   ecx, nb

      cmp   ecx, 3
      jle   @@tail3

      mov   eax, ecx
      shr   eax, 2
      and   ecx, 3

      shufps xmm2, xmm2, 0

   @@loop4:
      movaps   xmm0, [edi]
      movaps   xmm1, [edx]
      mulps    xmm1, xmm2
      addps    xmm0, xmm1
      movaps   [edi], xmm0

      add   edi, 16
      add   edx, 16
      dec   eax
      jnz   @@loop4

   @@tail3:
      test  ecx, ecx
      jz    @@done

   @@loop:
      movss xmm0, [edi]
      movss xmm1, [edx]
      mulss xmm1, xmm2
      addss xmm0, xmm1
      movss [edi], xmm0
      add   edi, 4
      add   edx, 4
      dec   ecx
      jnz   @@loop

   @@done:
      pop   edi
end;
(*asm
      fld   scale
      test  ecx, ecx
      jz    @@done

   @@loop:
      fld   dword ptr [eax + ecx*4 - 4]
      fld   dword ptr [edx + ecx*4 - 4]
      fmul  st(0), st(2)
      faddp
      fstp  dword ptr [eax + ecx*4 - 4]
      dec   ecx
      jnz   @@loop

   @@done:
      fstp st(0)
end;*)
{$else}
type
   TSingleArray = array [0..MaxInt shr 3] of Single;
   PSingleArray = ^TSingleArray;
begin
   while nb > 0 do begin
      PSingleArray(pTarget)[nb-1] := PSingleArray(pTarget)[nb-1] + PSingleArray(pOperand)[nb-1] * scale;
      Dec(nb);
   end;
end;
{$endif}

// daNNSubtract
//
function daNNSubtract(const left, right : ISingleArray) : ISingleArray;
begin
   var n := left.Length;
   Assert(right.Length = n);
   Result := NewSingleArray(n);
   var pLeft := left.Ptr;
   var pRight := right.Ptr;
   var pResult := Result.Ptr;
   for var i := 0 to n-1 do
      pResult[i] := pLeft[i] - pRight[i];
end;

// daNNMathSelfTest
//
procedure daNNMathSelfTest;
var
   p1, p2 : ISingleArray;

   procedure TestFloat(value, expected : Single; const prefix : String);
   begin
      Assert(Abs(value - expected) < 1e-5, Format('%sexpected %f but got %f', [ prefix, expected, value ]));
   end;

   procedure TestDotProduct(n : Integer; expected : Single);
   begin
      p1 := NewSingleArray(n);
      p2 := NewSingleArray(n);
      for var i := 0 to n-1 do begin
         p1[i] := 1 + i * 0.1;
         p2[i] := 2 + i * 0.3;
      end;
      var r := daNNDotProduct(p1.Ptr, p2.Ptr, n);
      TestFloat(r, expected, 'N = ' + IntToStr(n) + ', ');
   end;

begin
   TestDotProduct(0, 0);
   TestDotProduct(1, 2);
   TestDotProduct(2, 4.53);
   TestDotProduct(3, 7.65);
   TestDotProduct(4, 11.42);
   TestDotProduct(5, 15.9);
   TestDotProduct(6, 21.15);
   TestDotProduct(7, 27.23);
   TestDotProduct(8, 34.20);
   TestDotProduct(9, 42.12);
   TestDotProduct(10, 51.05);
   TestDotProduct(11, 61.05);
   TestDotProduct(12, 72.18);
   TestDotProduct(13, 84.5);

   daNNAddScaledOperand(p1.Ptr, p2.Ptr, 3, 0);
   TestFloat(p1[0], 1, 'scale a0 ');
   daNNAddScaledOperand(p1.Ptr, p2.Ptr, 3, 1);
   TestFloat(p1[0], 7, 'scale b0 ');
   TestFloat(p1[1], 1.1, 'scale b1 ');
   daNNAddScaledOperand(p1.Ptr, p2.Ptr, 3, 2);
   TestFloat(p1[0], 13, 'scale c0 ');
   TestFloat(p1[1], 8, 'scale c1 ');
   TestFloat(p1[2], 1.2, 'scale c2 ');
   daNNAddScaledOperand(p1.Ptr, p2.Ptr, 3, 5);
   TestFloat(p1[0], 19, 'scale d0 ');
   TestFloat(p1[1], 14.9, 'scale d1 ');
   TestFloat(p1[2], 9, 'scale d2 ');
   TestFloat(p1[3], 10, 'scale d3 ');
   TestFloat(p1[4], 11, 'scale d4 ');
   TestFloat(p1[5], 1.5, 'scale d5 ');
end;

type
   TdaNNSingleArray = class (TInterfacedObject, ISingleArray)
      private
         FBuffer : Pointer;    // base of allocated buffer
         FData : PSingleArray; // base of highly-aligned array
         FCount : Integer;

      public
         constructor Create(aCount : Integer);
         destructor Destroy; override;

         function Length : Integer;
         function High : Integer;
         function GetItem(index : Integer) : Single;
         procedure SetItem(index : Integer; v : Single);
         property Items[index : Integer] : Single read GetItem write SetItem; default;
         function Ptr : PSingleArray;

         function SumOfSquares : Double;
   end;

// NewSingleArray
//
function NewSingleArray(size : Integer) : ISingleArray;
begin
   Result := TdaNNSingleArray.Create(size);
end;

// Create
//
constructor TdaNNSingleArray.Create(aCount : Integer);
begin
   inherited Create;
   FCount := aCount;
   FBuffer := AllocMem(aCount*SizeOf(Single)+32);
   FData := Pointer((NativeUInt(FBuffer) + 31) and (NativeUInt(-1) - $1F));
end;

// Destroy
//
destructor TdaNNSingleArray.Destroy;
begin
   inherited;
   FreeMem(FBuffer);
end;

// Length
//
function TdaNNSingleArray.Length : Integer;
begin
   Result := FCount;
end;

// High
//
function TdaNNSingleArray.High : Integer;
begin
   Result := FCount-1;
end;

// GetItem
//
function TdaNNSingleArray.GetItem(index : Integer) : Single;
begin
   Assert(NativeUInt(index) < NativeUInt(FCount));
   Result := FData[index];
end;

// SetItem
//
procedure TdaNNSingleArray.SetItem(index : Integer; v : Single);
begin
   Assert(NativeUInt(index) < NativeUInt(FCount));
   FData[index] := v;
end;

// Ptr
//
function TdaNNSingleArray.Ptr : PSingleArray;
begin
   Result := FData;
end;

// SumOfSquares
//
function TdaNNSingleArray.SumOfSquares : Double;
begin
   Result := 0;
   for var i := 0 to FCount-1 do
      Result := Result + FData[i];
end;

type
   TdaNNSingleMatrix = class (TInterfacedObject, ISingleMatrix)
      private
         FBuffer : Pointer;    // base of allocated buffer
         FData : PSingleArray; // base of highly-aligned array
         FRowCount, FColCount : Integer;
         FColAlignedCount : Integer;

         FCompiledDotProduct : IdaNNJIT;

      public
         constructor Create(colCount, rowCount : Integer);
         destructor Destroy; override;

         function ColumnCount : Integer;
         function AlignedColumnCount : Integer;
         function RowCount : Integer;
         function Count : Integer;
         function GetItem(col, row : Integer) : Single;
         procedure SetItem(col, row : Integer; v : Single);
         property Items[col, row : Integer] : Single read GetItem write SetItem; default;
         function RowPtr(row : Integer) : PSingleArray;

         procedure MultiplyVector(const vector, result : ISingleArray);
   end;

// NewSingleMatrix
//
function NewSingleMatrix(colCount, rowCount : Integer) : ISingleMatrix;
begin
   Result := TdaNNSingleMatrix.Create(colCount, rowCount);
end;

// Create
//
constructor TdaNNSingleMatrix.Create(colCount, rowCount : Integer);
begin
   inherited Create;
   FColCount := colCount;
   FRowCount := rowCount;
   FColAlignedCount := FColCount + (8 - (FColCount and 7));

   FBuffer := AllocMem(FRowCount * FColAlignedCount * SizeOf(Single) + 32);
   FData := Pointer((NativeUInt(FBuffer) + 31) and (NativeUInt(-1) - $1F));

   FCompiledDotProduct := CompileDotProduct(ColumnCount);
end;

// Destroy
//
destructor TdaNNSingleMatrix.Destroy;
begin
   inherited;
   FreeMem(FBuffer);
end;

// ColumnCount
//
function TdaNNSingleMatrix.ColumnCount : Integer;
begin
   Result := FColCount;
end;

// AlignedColumnCount
//
function TdaNNSingleMatrix.AlignedColumnCount : Integer;
begin
   Result := FColAlignedCount;
end;

// RowCount
//
function TdaNNSingleMatrix.RowCount : Integer;
begin
   Result := FRowCount;
end;

// Count
//
function TdaNNSingleMatrix.Count : Integer;
begin
   Result := FColCount * FRowCount;
end;

// GetItem
//
function TdaNNSingleMatrix.GetItem(col, row : Integer) : Single;
begin
   Result := FData[col + row * FColAlignedCount];
end;

// SetItem
//
procedure TdaNNSingleMatrix.SetItem(col, row : Integer; v : Single);
begin
   FData[col + row * FColAlignedCount] := v;
end;

// RowPtr
//
function TdaNNSingleMatrix.RowPtr(row : Integer) : PSingleArray;
begin
   Assert(NativeUInt(row) < NativeUInt(FRowCount));
   Result := @FData[ row * FColAlignedCount ];
end;

// daNNDotProduct3
//
procedure daNNDotProduct3(pVec, pMat : PSingleArray; vecSize, matStride : Integer; dest : PSingleArray);
// pVec -> rcx
// pMat -> rdx
// vecSize -> r8
// matStride -> r9
{$ifdef WIN64_ASM}
asm
      pxor  xmm0, xmm0
      pxor  xmm1, xmm1
      pxor  xmm2, xmm2

      cmp   r8d, 3 + 4  // a single loop4 is not beneficial
      jle   @@tail3

      mov   eax, r8d
      shr   eax, 2
      and   r8d, 3

   @@loop4:
      movaps   xmm7, [rcx]

      movaps   xmm3, [rdx]
      mulps    xmm3, xmm7
      addps    xmm0, xmm3

      movaps   xmm4, [rdx + r9]
      mulps    xmm4, xmm7
      addps    xmm1, xmm4

      movaps   xmm5, [rdx + 2*r9]
      mulps    xmm5, xmm7
      addps    xmm2, xmm5

      add   rcx, 16
      add   rdx, 16
      dec   eax
      jnz   @@loop4

      movshdup xmm3, xmm0
      addps xmm0, xmm3
      movhlps xmm3, xmm0
      addss xmm0, xmm3

      movshdup xmm4, xmm1
      addps xmm1, xmm4
      movhlps xmm4, xmm1
      addss xmm1, xmm4

      movshdup xmm5, xmm2
      addps xmm2, xmm5
      movhlps xmm5, xmm2
      addss xmm2, xmm5

   @@tail3:
      test  r8d, r8d
      jz    @@done

   @@loop:
      movss xmm7, [rcx]

      movss xmm3, [rdx]
      mulss xmm3, xmm7
      addss xmm0, xmm3

      movss xmm4, [rdx + r9]
      mulss xmm4, xmm7
      addss xmm1, xmm4

      movss xmm5, [rdx + 2*r9]
      mulss xmm5, xmm7
      addss xmm2, xmm5

      add   rcx, 4
      add   rdx, 4
      dec   r8d
      jnz   @@loop

   @@done:
      mov   rax, dest
      movss [rax], xmm0
      movss [rax+4], xmm1
      movss [rax+8], xmm2
end;
{$endif}

// MultiplyVector
//
procedure TdaNNSingleMatrix.MultiplyVector(const vector, result : ISingleArray);
begin
   Assert(vector.Length = ColumnCount, 'Vector size mismatches matrix row count');
   Assert(result.Length = RowCount, 'Result size mismatches matrix column count');

   var resultPtr := result.Ptr;
   var vectorPtr := vector.Ptr;

   var cdp := FCompiledDotProduct.Ptr;

   for var row := 0 to RowCount-1 do
      resultPtr[row] := TCompiledDotProduct(cdp)(vectorPtr, @FData[ row * FColAlignedCount ]);
//      resultPtr[row] := daNNDotProduct(vectorPtr, @FData[ row * FColAlignedCount ], ColumnCount);
//}
{
   var row := 0;
   while row <= RowCount-3 do begin
      daNNDotProduct3(vectorPtr, @FData[ row * FColAlignedCount ], ColumnCount, FColAlignedCount*4, @resultPtr[row]);
      Inc(row, 3);
   end;
   while row < RowCount do begin
      resultPtr[row] := daNNDotProduct(vectorPtr, @FData[ row * FColAlignedCount ], ColumnCount);
      Inc(row);
   end; //}
end;

end.

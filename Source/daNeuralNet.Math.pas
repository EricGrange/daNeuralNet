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
{$define USE_CBLAS}

{$ifdef USE_CBLAS}
   {.$define USE_CBLAS_FOR_SGEMV}
   {.$define USE_CBLAS_FOR_SGEMM}
{$endif}

interface

uses SysUtils, Classes;

type
   PSingle = System.PSingle;  // workaround declaration conflict between WinApi.Windows and System

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
      procedure AddScaledVector(scale : Single; const vector : ISingleArray);
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
      function GetRowPtr(row : Integer) : PSingleArray;
      property RowPtr[row : Integer] : PSingleArray read GetRowPtr;

      procedure Multiply(const matrix, result : ISingleMatrix);

      procedure MultiplyVector(const vector, result : ISingleArray);
      procedure TransposeMultiplyVector(const vector, result : ISingleArray);

      procedure AddScaledVectorToRow(scale : Single; const vector : ISingleArray; row : Integer);
   end;

   TMatrixOption = ( moPacked );
   TMatrixOptions = set of TMatrixOption;

const
   cDefaultMatrixOptions = {$ifdef USE_CBLAS} [ moPacked ] {$else} [ ] {$endif};

function NewSingleArray(size : Integer) : ISingleArray;

function NewSingleMatrix(colCount, rowCount : Integer; options: TMatrixOptions = cDefaultMatrixOptions) : ISingleMatrix;

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

uses
   {$ifdef USE_CBLAS}
   LibCBLAS,
   {$endif}
   daNeuralNet.JIT
   ;

// daNNDotProduct
//
function daNNDotProduct(p1, p2 : PSingleArray; nb : Integer) : Single;
{$ifdef WIN64_ASM}
asm
      pxor  xmm0, xmm0
      cmp   r8d, 3 + 4  // a single loop4 is not beneficial
      jle   @@tail3

      mov   eax, r8d
      shr   eax, 2
      and   r8d, 3

   @@loop4:
      movups   xmm1, [rcx]
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
{$ifdef WIN64_ASM}
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
         procedure AddScaledVector(scale : Single; const vector : ISingleArray);
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

// AddScaledVector
//
procedure TdaNNSingleArray.AddScaledVector(scale : Single; const vector : ISingleArray);
begin
   Assert(vector.Length = FCount, 'Mismatched vector length');

   {$ifdef USE_CBLAS}
   CBLAS.saxpy(FCount, scale, PSingle(vector.Ptr), 1, FBuffer, 1);
   {$else}
   daNNAddScaledOperand(FBuffer, vector.Ptr, scale, FCount);
   {$endif}
end;

type
   TdaNNSingleMatrix = class (TInterfacedObject, ISingleMatrix)
      private
         FBuffer : Pointer;    // base of allocated buffer
         FData : PSingleArray; // base of highly-aligned array
         FRowCount, FColCount : Integer;
         FColAlignedCount : Integer;
         {$ifndef USE_CBLAS_FOR_SGEMV}
         FCompiledMultVector : IdaNNJIT;
         {$endif}

      public
         constructor Create(colCount, rowCount : Integer; options: TMatrixOptions);
         destructor Destroy; override;

         function ColumnCount : Integer; inline;
         function AlignedColumnCount : Integer; inline;
         function RowCount : Integer; inline;
         function Count : Integer;

         function GetItem(col, row : Integer) : Single;
         procedure SetItem(col, row : Integer; v : Single);
         property Items[col, row : Integer] : Single read GetItem write SetItem; default;
         function GetRowPtr(row : Integer) : PSingleArray;

         procedure Multiply(const matrix, result : ISingleMatrix);

         procedure MultiplyVector(const vector, result : ISingleArray);
         procedure TransposeMultiplyVector(const vector, result : ISingleArray);
         procedure AddScaledVectorToRow(scale : Single; const vector : ISingleArray; row : Integer);
   end;

// NewSingleMatrix
//
function NewSingleMatrix(colCount, rowCount : Integer; options: TMatrixOptions = cDefaultMatrixOptions) : ISingleMatrix;
begin
   Result := TdaNNSingleMatrix.Create(colCount, rowCount, options);
end;

// Create
//
constructor TdaNNSingleMatrix.Create(colCount, rowCount : Integer; options: TMatrixOptions);
begin
   inherited Create;
   FColCount := colCount;
   FRowCount := rowCount;
   if moPacked in options then
      FColAlignedCount := FColCount
   else FColAlignedCount := FColCount + (8 - (FColCount and 7));

   FBuffer := AllocMem(FRowCount * FColAlignedCount * SizeOf(Single) + 32);
   FData := Pointer((NativeUInt(FBuffer) + 31) and (NativeUInt(-1) - $1F));

   {$ifndef USE_CBLAS_FOR_SGEMV}
   FCompiledMultVector := CompileMultVectorAVX(ColumnCount, RowCount);
   {$endif}
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

// GetRowPtr
//
function TdaNNSingleMatrix.GetRowPtr(row : Integer) : PSingleArray;
begin
   Assert(NativeUInt(row) < NativeUInt(FRowCount));
   Result := @FData[ row * FColAlignedCount ];
end;

// Multiply
//
procedure TdaNNSingleMatrix.Multiply(const matrix, result : ISingleMatrix);
begin
   Assert(matrix.RowCount = ColumnCount, 'Matrix row count mismatch');
   Assert(result.ColumnCount = matrix.ColumnCount, 'result column count mismatch');
   Assert(result.RowCount = RowCount, 'result column count mismatch');

   {$ifdef USE_CBLAS_FOR_SGEMM}

   cblas.sgemm(
      cblasRowMajor, cblasNoTrans, cblasNoTrans,
      RowCount, matrix.ColumnCount, ColumnCount,         // m, n, k
      1, PSingle(FData), ColumnCount,                    // alpha, A, lda
      PSingle(matrix.RowPtr[0]), matrix.ColumnCount,     // B, ldb
      0, PSingle(result.RowPtr[0]), matrix.ColumnCount   // beta, C, ldc
   );

   {$else}

   var kSize := matrix.AlignedColumnCount;
   var r : Double;
   for var i := 0 to RowCount-1 do begin
      var pMat1i := GetRowPtr(i);
      var pMatRi := result.RowPtr[i];
      for var j := 0 to matrix.ColumnCount-1 do begin
         var pMat2j := PSingleArray(@matrix.RowPtr[0][j]);
         r := pMat1i[0] * pMat2j[0];
         for var k := 1 to ColumnCount-1 do begin
            pMat2j := @pMat2j[kSize];
            r := r + pMat1i[k] * pMat2j[0];
         end;
         pMatRi[j] := r;
      end;
   end;

   {$endif}
end;

// MultiplyVector
//
procedure TdaNNSingleMatrix.MultiplyVector(const vector, result : ISingleArray);
begin
   Assert(vector.Length = ColumnCount, 'Vector size mismatches matrix column count');
   Assert(result.Length = RowCount, 'Result size mismatches matrix row count');

   var resultPtr := result.Ptr;
   var vectorPtr := vector.Ptr;

   {$ifdef USE_CBLAS_FOR_SGEMV}

   cblas.sgemv(cblasRowMajor, cblasNoTrans,
               RowCount, ColumnCount, 1.0, // m, n, alpha
               @FData[0], ColumnCount,   // a, lda
               PSingle(vectorPtr), 1, // x, incX
               0, // beta
               PSingle(resultPtr), 1 // y, incY
               );

   {$else}

   var cdp := FCompiledMultVector.Ptr;

   TCompiledMatVectorMult(cdp)(@FData[0], vectorPtr, resultPtr);
   {$endif}
end;

// TransposeMultiplyVector
//
procedure TdaNNSingleMatrix.TransposeMultiplyVector(const vector, result : ISingleArray);
begin
   Assert(vector.Length = RowCount, 'Vector size mismatches matrix row count');
   Assert(result.Length = ColumnCount, 'Result size mismatches matrix column count');

   var resultPtr := result.Ptr;
   var vectorPtr := vector.Ptr;

   {$ifdef USE_CBLAS}

   cblas.sgemv(cblasRowMajor, cblasTrans,
               RowCount, ColumnCount, 1.0, // m, n, alpha
               @FData[0], ColumnCount,   // a, lda
               PSingle(vectorPtr), 1, // x, incX
               0, // beta
               PSingle(resultPtr), 1 // y, incY
               );

   {$else}
   for var col := 0 to ColumnCount-1 do begin
      var accum : Double;
      accum := 0;
      for var row := 0 to RowCount-1 do
         accum := accum + vectorPtr[row] * GetItem(col, row);
      resultPtr[col] := accum;
   end;
   {$endif}
end;

// AddScaledVectorToRow
//
procedure TdaNNSingleMatrix.AddScaledVectorToRow(scale : Single; const vector : ISingleArray; row : Integer);
begin
   Assert(vector.Length = FColCount, 'Mismatched vector length');

   var rowPtr := GetRowPtr(row);

   {$ifdef USE_CBLAS}
   CBLAS.saxpy(FColCount, scale, PSingle(vector.Ptr), 1, PSingle(rowPtr), 1);
   {$else}
   daNNAddScaledOperand(rowPtr, vector.Ptr, scale, FColCount);
   {$endif}
end;

// ------------------------------------------------------------------
// ------------------------------------------------------------------
// ------------------------------------------------------------------
initialization
// ------------------------------------------------------------------
// ------------------------------------------------------------------
// ------------------------------------------------------------------

   {$ifdef USE_CBLAS}
   LoadLibCBLAS;
   CBLAS.openblas_set_num_threads(1);
   {$endif}

end.

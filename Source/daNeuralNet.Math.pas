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

uses SysUtils, daNeuralNet;

function  daNNDotProduct(p1, p2 : PSingle; nb : Integer) : Single;
procedure daNNAddScaledOperand(pTarget, pOperand : PSingle; scale : Single; nb : Integer);
function  daNNSubtract(left, right : TdaNNSingleArray) : TdaNNSingleArray;

procedure daNNMathSelfTest;

// ------------------------------------------------------------------
// ------------------------------------------------------------------
// ------------------------------------------------------------------
implementation
// ------------------------------------------------------------------
// ------------------------------------------------------------------
// ------------------------------------------------------------------

// daNNDotProduct
//
function daNNDotProduct(p1, p2 : PSingle; nb : Integer) : Single;
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

      pshufd   xmm1, xmm0, $E
      addps    xmm0, xmm1
      pshufd   xmm1, xmm0, $1
      addss    xmm0, xmm1

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
      movups   xmm1, [eax]
      movups   xmm2, [edx]
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
procedure daNNAddScaledOperand(pTarget, pOperand : PSingle; scale : Single; nb : Integer);
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
function daNNSubtract(left, right : TdaNNSingleArray) : TdaNNSingleArray;
begin
   var n := Length(left);
   Assert(Length(right) = n);
   SetLength(Result, n);
   for var i := 0 to n-1 do
      Result[i] := left[i] - right[i];
end;

// daNNMathSelfTest
//
procedure daNNMathSelfTest;
var
   p1, p2 : TdaNNSingleArray;

   procedure TestFloat(value, expected : Single; const prefix : String);
   begin
      Assert(Abs(value - expected) < 1e-5, Format('%sexpected %f but got %f', [ prefix, expected, value ]));
   end;

   procedure TestDotProduct(n : Integer; expected : Single);
   begin
      SetLength(p1, n);
      SetLength(p2, n);
      for var i := 0 to n-1 do begin
         p1[i] := 1 + i * 0.1;
         p2[i] := 2 + i * 0.3;
      end;
      var r := daNNDotProduct(PSingle(p1), PSingle(p2), n);
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

   daNNAddScaledOperand(PSingle(p1), PSingle(p2), 3, 0);
   TestFloat(p1[0], 1, 'scale a0 ');
   daNNAddScaledOperand(PSingle(p1), PSingle(p2), 3, 1);
   TestFloat(p1[0], 7, 'scale b0 ');
   TestFloat(p1[1], 1.1, 'scale b1 ');
   daNNAddScaledOperand(PSingle(p1), PSingle(p2), 3, 2);
   TestFloat(p1[0], 13, 'scale c0 ');
   TestFloat(p1[1], 8, 'scale c1 ');
   TestFloat(p1[2], 1.2, 'scale c2 ');
   daNNAddScaledOperand(PSingle(p1), PSingle(p2), 3, 5);
   TestFloat(p1[0], 19, 'scale d0 ');
   TestFloat(p1[1], 14.9, 'scale d1 ');
   TestFloat(p1[2], 9, 'scale d2 ');
   TestFloat(p1[3], 10, 'scale d3 ');
   TestFloat(p1[4], 11, 'scale d4 ');
   TestFloat(p1[5], 1.5, 'scale d5 ');
end;

end.

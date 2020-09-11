{**********************************************************************}
{                                                                      }
{    "The contents of this file are subject to the Mozilla Public      }
{    License Version 2.0 (the "License"); you may not use this         }
{    file except in compliance with the License. You may obtain        }
{    a copy of the License at                                          }
{                                                                      }
{    http://www.mozilla.org/MPL/                                       }
{                                                                      }
{    Software distributed under the License is distributed on an       }
{    "AS IS" basis, WITHOUT WARRANTY OF ANY KIND, either express       }
{    or implied. See the License for the specific language             }
{    governing rights and limitations under the License.               }
{                                                                      }
{**********************************************************************}

// danNN include file

{global compiler options}

{$A+} // align on
{$B-} // BoolEval off
{$X+} // extended syntax
{$H+} // long strings
{$Q-} // overflow checks off
{$J-} // assignable typed constants off

{$EXCESSPRECISION OFF}

{$WARN DUPLICATE_CTOR_DTOR OFF}

{$IFDEF FPC}
   {$mode delphi}
   {$DEFINE PUREPASCAL}
   {$ASMMODE INTEL}
{$ENDIF}

{.$DEFINE PUREPASCAL}

{$IFNDEF PUREPASCAL}
   {$IF Defined(WIN32)}
      {$DEFINE WIN32_ASM}
   {$ELSEIF Defined(WIN64)}
      {$DEFINE WIN64_ASM}
   {$IFEND}
{$ENDIF}



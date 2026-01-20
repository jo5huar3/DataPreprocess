{-# LANGUAGE OverloadedStrings #-}

module Main where

import qualified Data.Text.IO as T
import System.Directory (createDirectoryIfMissing)
import System.FilePath ((</>))
import Control.Monad (forM_)
import Text.Printf (printf)

import System.Environment (getArgs)
import Data.Maybe (mapMaybe)
import Data.List (intersect, intercalate, isInfixOf, isPrefixOf)
import qualified Data.Set as Set
import qualified Data.Map.Strict as Map
import qualified Data.Text as Text
import Data.Char (isSpace)     
import Data.List (sortOn) 
import Cryptol.Parser
  ( parseModule
  , defaultConfig
  , Config(..)
  )

import Cryptol.Parser.AST
  ( Module
  , PName
  , TopDecl(..)
  , Decl(..)
  , TopLevel(..)
  , Bind(..)
  , BindImpl(..)
  , Schema(..)
  , TySyn(..)
  , EnumCon(..)
  , Type(..)
  , Prop(..)
  , Named(..)
  , Newtype(..)
  , EnumDecl(..)
  , PrimType(..)
  , Expr(..)
  , Match(..)
  , CaseAlt(..)
  , UpdField(..)
  , UpdHow(..)
  , PropGuardCase(..)
  , Pattern(..)
  , mDecls
  , tsName
  , primTName
  , primTCts
  , nBody
  , nName
  , eName
  , eCons
  , bindImpl
  , bindParams
  )

import Cryptol.Parser.Position (thing)
import Cryptol.Utils.PP (pretty)
import Cryptol.Utils.RecordMap (recordElements)


--------------------------------------------------------------------------------
-- main
--------------------------------------------------------------------------------

main :: IO ()
main = do
  args <- getArgs
  let (path, mOutDir) =
        case args of
          (p:out:_) -> (p, Just out)   -- INPUT + OUT_DIR
          (p:_)     -> (p, Nothing)    -- only INPUT
          []        -> ("Toy.cry", Nothing)

  src <- T.readFile path

  let cfg :: Config
      cfg = defaultConfig { cfgSource = path }

  case parseModule cfg src of
    Left err -> do
      putStrLn "Parse error:"
      print err

    Right mods -> do
      let srcStr = Text.unpack src
      putStrLn $ "Parsed " ++ show (length mods) ++ " module(s)."
      case mOutDir of
        Nothing     -> mapM_ dumpModule mods
        Just outDir -> mapM_ (writeModuleSnippets outDir srcStr) mods

--------------------------------------------------------------------------------
-- Types & small helpers
--------------------------------------------------------------------------------

type NameSet = Set.Set PName
type NameMap = Map.Map PName [TopDecl PName]

-- All top-level declarations for a module
topDecls :: Module PName -> [TopDecl PName]
topDecls = mDecls

-- Pick a “root” name for a definitional TopDecl (first name it defines)
rootNameFromTopDecl :: TopDecl PName -> Maybe PName
rootNameFromTopDecl td =
  case defNamesFromTopDecl td of
    (n:_) -> Just n
    []    -> Nothing

--------------------------------------------------------------------------------
-- Stage 1: dump to stdout (for debugging)
--------------------------------------------------------------------------------

dumpModule :: Module PName -> IO ()
dumpModule m = do
  putStrLn "================================"
  putStrLn "Definitions in this module:"
  let decls   = topDecls m
      nameMap = buildNameMap decls

      roots = [ td | td <- decls, isValueRootTopDecl td ]

  forM_ roots $ \td -> do
    case rootNameFromTopDecl td of
      Nothing -> pure ()
      Just nm -> do
        let slice0 = sliceFor nameMap decls nm
            slice  = normalizeSliceOrder decls slice0
        putStrLn "--------------------------------"
        putStrLn ("Slice for: " ++ pretty nm)
        putStrLn (unlines [ prettyTopDecl d | d <- slice ])

--------------------------------------------------------------------------------
-- Stage 2: actual file writing
--------------------------------------------------------------------------------

writeModuleSnippets :: FilePath -> String -> Module PName -> IO ()
writeModuleSnippets outDir srcStr m = do
  let decls   = topDecls m
      nameMap = buildNameMap decls

      roots :: [TopDecl PName]
      roots = [ td | td <- decls, isValueRootTopDecl td ]

      -- Get all import lines from the original source text
      importLines = importLinesFromSource srcStr
      importBlock =
        if null importLines
           then ""
           else unlines importLines

  createDirectoryIfMissing True outDir

  forM_ (zip [1 :: Int ..] roots) $ \(i, td) ->
    case rootNameFromTopDecl td of
      Nothing -> pure ()
      Just nm -> do
        let slice0   = sliceFor nameMap decls nm
            slice    = normalizeSliceOrder decls slice0
            baseName = pretty nm
            fileName = printf "%03d_%s.cry" i baseName
            outPath  = outDir </> fileName

            bodyBlock = unlines [ prettyTopDecl d | d <- slice ]

            contents =
              if null importBlock
                 then bodyBlock
                 else importBlock ++ "\n\n" ++ bodyBlock

        writeFile outPath contents

--------------------------------------------------------------------------------
-- Core logic: given a root name, compute its dependency closure
--------------------------------------------------------------------------------

-- | Build a map from name -> all top-level declarations that define it.
buildNameMap :: [TopDecl PName] -> NameMap
buildNameMap decls =
  Map.fromListWith (++)
    [ (nm, [td])
    | td <- decls
    , nm <- defNamesFromTopDecl td
    ]

-- | One step of the graph: “what names does this top-level mention?”
refsOfTopDecl :: TopDecl PName -> NameSet
refsOfTopDecl td =
  case td of
    Decl tl -> namesInDecl (tlValue tl)
    _       -> Set.empty

-- | Compute the transitive closure of names reachable from a root name,
--   staying *within this module* (only follow names that have a TopDecl here).
closureOf :: NameMap -> (TopDecl PName -> NameSet) -> PName -> NameSet
closureOf nameMap refsOf root =
  go Set.empty (Set.singleton root)
  where
    go seen pending =
      case Set.minView pending of
        Nothing -> seen
        Just (nm, rest) ->
          if nm `Set.member` seen
            then go seen rest
            else
              let declsFor = Map.findWithDefault [] nm nameMap
                  deps     = Set.unions (map refsOf declsFor)
                  depsLocal = Set.filter (`Map.member` nameMap) deps
                  pending'  = rest `Set.union` depsLocal
              in go (Set.insert nm seen) pending'

-- | Given a root name and the whole module, return the slice:
--   every TopDecl whose defined names intersect the reachable set.
sliceFor :: NameMap -> [TopDecl PName] -> PName -> [TopDecl PName]
sliceFor nameMap decls rootNm =
  let reachable :: NameSet
      reachable = closureOf nameMap refsOfTopDecl rootNm
  in
    [ td
    | td <- decls
    , let defs = defNamesFromTopDecl td
    , not (null (defs `intersect` Set.toList reachable))
    ]

-- Remove trailing newline characters, but leave other whitespace alone.
trimTrailingNewlines :: String -> String
trimTrailingNewlines =
  reverse . dropWhile (== '\n') . reverse


--------------------------------------------------------------------------------
-- “Defined names” for each TopDecl
-- (values, properties, type synonyms, etc.)
--------------------------------------------------------------------------------

defNamesFromTopDecl :: TopDecl PName -> [PName]
defNamesFromTopDecl td =
  case td of
    Decl tl      -> defNamesFromDecl (tlValue tl)
    -- If you want: add DPrimType, TDNewtype, TDEnum here later
    _            -> []

defNamesFromDecl :: Decl PName -> [PName]
defNamesFromDecl d =
  case d of
    DBind b         -> [ thing (bName b) ]
    DRec bs         -> [ thing (bName b) | b <- bs ]
    DSignature ns _ -> map thing ns             -- type signatures
    DType ts        -> [ thing (tsName ts) ]    -- type synonyms
    DPatBind p _    ->
      case p of
        PVar x      -> [ thing x ]             -- sbox = [...]
        PLocated p' _ -> defNamesFromDecl (DPatBind p' undefined)
        _           -> []
    DLocated d' _   -> defNamesFromDecl d'
    _               -> []

--------------------------------------------------------------------------------
-- “Referenced names” inside decls (value + type level)
--------------------------------------------------------------------------------

-- | Names that appear in schemas (constraints + result type).
namesInSchema :: Schema PName -> NameSet
namesInSchema (Forall _ props ty _) =
  namesInType ty `Set.union` Set.unions (map namesInProp props)

namesInProp :: Prop PName -> NameSet
namesInProp (CType ty) = namesInType ty

-- | Type-level names (e.g. GF28 in GF28 -> GF28)
namesInType :: Type PName -> NameSet
namesInType ty =
  case ty of
    TFun a b       -> namesInType a `Set.union` namesInType b
    TSeq a b       -> namesInType a `Set.union` namesInType b
    TBit           -> Set.empty
    TNum _         -> Set.empty
    TChar _        -> Set.empty
    TUser n args   -> Set.insert (thing n) (Set.unions (map namesInType args))
    TTyApp ts      -> Set.unions [ namesInType (value t) | t <- ts ]
    TRecord _      -> Set.empty
    TTuple ts      -> Set.unions (map namesInType ts)
    TWild          -> Set.empty
    TLocated t _   -> namesInType t
    TParens t _    -> namesInType t
    TInfix a _ _ b -> namesInType a `Set.union` namesInType b

-- | All names used anywhere in this value-level declaration:
--   - variables in expressions
--   - type names in schemas & synonyms
namesInDecl :: Decl PName -> NameSet
namesInDecl d =
  case d of
    DBind b ->
      case bindImpl b of
        Just (DExpr e)        -> namesInExpr e
        Just (DPropGuards cs) -> Set.unions [ namesInExpr (pgcExpr c) | c <- cs ]
        Nothing               -> Set.empty

    DRec bs ->
      Set.unions [ namesInDecl (DBind b) | b <- bs ]

    DPatBind _ e ->
      namesInExpr e

    DSignature _ s ->
      namesInSchema s

    -- ⬇️ FIXED HERE
    DType (TySyn _ _ _ ty) ->
      namesInType ty

    DLocated d' _ ->
      namesInDecl d'

    _ ->
      Set.empty


-- | Names used in an expression (Cryptol AST).
--   (This is the same idea you already had; only EVar introduces a name.)
namesInExpr :: Expr PName -> NameSet
namesInExpr expr =
  case expr of
    EVar n        -> Set.singleton n
    ELit _        -> Set.empty

    EGenerate e   -> go e
    ETuple es     -> Set.unions (map go es)
    ERecord rec   -> Set.unions [ go e | (_, e) <- recordElements rec ]
    ESel e _      -> go e

    EUpd me fields ->
      maybe Set.empty go me
      `Set.union` Set.unions [ go e | UpdField _ _ e <- fields ]

    EList es      -> Set.unions (map go es)

    -- ranges: numeric/type-level, ignore
    EFromTo{}         -> Set.empty
    EFromToBy{}       -> Set.empty
    EFromToDownBy{}   -> Set.empty
    EFromToLessThan{} -> Set.empty

    EInfFrom e me ->
      go e `Set.union` maybe Set.empty go me

    EComp e mss ->
      go e `Set.union`
      Set.unions [ Set.unions (map namesInMatch ms) | ms <- mss ]

    EApp e1 e2   -> go e1 `Set.union` go e2
    EAppT e _    -> go e
    EIf e1 e2 e3 -> go e1 `Set.union` go e2 `Set.union` go e3

    ECase e alts ->
      go e `Set.union`
      Set.unions [ go rhs | CaseAlt _ rhs <- alts ]

    EWhere e ds  ->
      go e `Set.union` Set.unions (map namesInDecl ds)

    ETyped e _   -> go e
    ETypeVal _   -> Set.empty

    EFun _ _ e   -> go e
    ELocated e _ -> go e
    ESplit e     -> go e
    EParens e    -> go e

    EInfix e1 op _ e2 ->
      go e1 `Set.union` go e2 `Set.union` Set.singleton (thing op)

    EPrefix _ e  -> go e
  where
    go = namesInExpr

namesInMatch :: Match PName -> NameSet
namesInMatch m =
  case m of
    Match _ e  -> namesInExpr e
    MatchLet b -> namesInDecl (DBind b)

normalizeSliceOrder :: [TopDecl PName] -> [TopDecl PName] -> [TopDecl PName]
normalizeSliceOrder fullDecls slice =
  let firstOcc :: Map.Map PName Int
      firstOcc =
        Map.fromListWith min
          [ (nm, i)
          | (i, td) <- zip [0 :: Int ..] fullDecls
          , nm <- defNamesFromTopDecl td
          ]

      groupPos :: TopDecl PName -> Int
      groupPos td =
        case mapMaybe (`Map.lookup` firstOcc) (defNamesFromTopDecl td) of
          [] -> maxBound
          xs -> minimum xs

      sigPri :: TopDecl PName -> Int
      sigPri td = if isSignatureTopDecl td then 0 else 1
  in
    -- include original index to preserve stability even if keys tie
    map snd $
      sortOn (\(i, td) -> (groupPos td, sigPri td, i)) (zip [0 :: Int ..] slice)

-- Indent multi-line type signatures so that all continuation lines
-- are more indented than the first line.
--
-- Example input from 'pretty':
--
--   RCs : {w, n}
--   (fin w, fin n, 24 >= n, n == 12 + 2 * (lg2 w)) =>
--   [n][5][5][w]
--
-- becomes:
--
--   RCs : {w, n}
--     (fin w, fin n, 24 >= n, n == 12 + 2 * (lg2 w)) =>
--       [n][5][5][w]
--
indentSignatureLayout :: String -> String
indentSignatureLayout s =
  case lines s of
    []       -> ""
    (l : ls) -> unlines (l : go False ls)
  where
    go :: Bool -> [String] -> [String]
    go _ [] = []
    go sawArrow (ln:rest)
      -- ignore completely blank lines, just keep them as-is
      | all isSpace ln = ln : go sawArrow rest
      | otherwise =
          let indentWidth = if sawArrow then 4 else 2
              indent      = replicate indentWidth ' '
              ln'         = indent ++ ln
              sawArrow'   = sawArrow || "=>" `isInfixOf` ln
          in ln' : go sawArrow' rest

isValueRootTopDecl :: TopDecl PName -> Bool
isValueRootTopDecl td =
  case td of
    Decl tl ->
      case dropLocDecl (tlValue tl) of
        DBind{}    -> True
        DRec{}     -> True
        DPatBind{} -> True
        _          -> False
    _ -> False


isSignatureTopDecl :: TopDecl PName -> Bool
isSignatureTopDecl td =
  case td of
    Decl tl ->
      case dropLocDecl (tlValue tl) of
        DSignature{} -> True
        _            -> False
    _ -> False

-- Strip source-location wrappers so we can pattern match on the real Decl.
dropLocDecl :: Decl PName -> Decl PName
dropLocDecl (DLocated d _) = dropLocDecl d
dropLocDecl d              = d

-- Drop parser location wrappers like:
--   (at ../cryptol-specs/Common/EC/PrimeField/PFEC.cry:149:5--149:13, Infinity)
-- so they become just:
--   Infinity
--
-- We do this textually, line-by-line, preserving indentation.
stripAtAnnotations :: String -> String
stripAtAnnotations = unlines . map stripLine . lines
  where
    stripLine :: String -> String
    stripLine [] = []
    stripLine s@(c:cs)
      -- If the line *at this position* starts with "(at ",
      -- remove "(at ... , " and the closing ")" and keep only the identifier.
      | "(at " `isPrefixOf` s =
          let afterAt     = drop 4 s                     -- drop "(at "
              afterComma  = drop 1 (dropWhile (/= ',') afterAt)
              afterSpaces = dropWhile isSpace afterComma
              ident       = takeWhile (\ch -> ch /= ')' && not (isSpace ch)) afterSpaces
              afterParen  = dropWhile (/= ')') afterSpaces
              rest        = case afterParen of
                              []     -> ""
                              (_:xs) -> xs              -- drop ')'
          in ident ++ stripLine rest
      -- Otherwise, keep the first char and keep scanning.
      | otherwise = c : stripLine cs


-- Custom pretty-print for top-level declarations:
--  • if it's a constant/sequence = [ ... ] we print it on one line
--  • otherwise we fall back to the standard Cryptol pretty printer.
prettyTopDecl :: TopDecl PName -> String
prettyTopDecl td =
  trimTrailingNewlines $
    case td of
      -- Value-level declarations
      Decl tl ->
        let d = dropLocDecl (tlValue tl)

            ppDefault :: String
            ppDefault = fixCaseLayout (stripAtAnnotations (pretty td))
        in case d of

          -- PP = [ ... ]  -- no arguments, pure sequence
          DBind b
            | null (bindParams b) ->
                case bindImpl b of
                  Just (DExpr (EList es)) ->
                    let nmStr   = pretty (thing (bName b))
                        elems   = intercalate ", " [ pretty e | e <- es ]
                        txt     = nmStr ++ " = [" ++ elems ++ "]"
                    in fixCaseLayout (stripAtAnnotations txt)
                  _ ->
                    ppDefault

          -- sbox = [ ... ]  (pattern binding)
          DPatBind (PVar x) (EList es) ->
            let nmStr = pretty (thing x)
                elems = intercalate ", " [ pretty e | e <- es ]
                txt   = nmStr ++ " = [" ++ elems ++ "]"
            in fixCaseLayout (stripAtAnnotations txt)

          -- Type signatures: pretty-print, then fix annotations/case layout,
          -- then re-indent the multi-line signature.
          DSignature _ _ ->
            indentSignatureLayout ppDefault

          -- everything else: delegate to the standard pretty printer
          _ ->
            ppDefault

      -- Non-Decl top-levels (type decls, pragmas, etc.)
      _ ->
        fixCaseLayout (stripAtAnnotations (pretty td))

moduleDefNames :: [TopDecl PName] -> NameSet
moduleDefNames decls =
  Set.fromList (concatMap defNamesFromTopDecl decls)

-- Detect whether a line of source code is an import.
isImportLine :: String -> Bool
isImportLine line =
  case dropWhile isSpace line of
    ('i':'m':'p':'o':'r':'t':' ':_) -> True
    _                              -> False

-- Extract all import lines from the original source text.
importLinesFromSource :: String -> [String]
importLinesFromSource srcStr =
  [ line
  | line <- lines srcStr
  , isImportLine line
  ]

-- Fix layout for `case ... of` blocks so that all alternatives
-- are indented more than the `case` line.
--
-- Example (bad):
--   case p1 of
--   Infinity ->
--     ...
--
-- becomes (good):
--   case p1 of
--     Infinity ->
--       ...
fixCaseLayout :: String -> String
fixCaseLayout s = unlines (go [] (map splitIndent (lines s)))
  where
    -- Split a line into (indent width, indent string, rest)
    splitIndent :: String -> (Int, String, String)
    splitIndent line =
      let (ind, rest) = span isSpace line
      in (length ind, ind, rest)

    go :: [Int] -> [(Int, String, String)] -> [String]
    go _ [] = []
    go stack ((i, ind, rest) : xs) =
      let stack1 = dropWhile (> i) stack      -- pop cases that we just dedented out of
          content = rest                      -- rest has no leading spaces
      in
        if isCaseLine content then
          -- Start of a new `case ... of` block at indent `i`
          (ind ++ content) : go (i : stack1) xs
        else
          case stack1 of
            -- Inside a `case` block: fix alternatives that are not indented
            -- more than the `case` line and look like `... ->`
            (ci : _) | i <= ci && looksLikeAlt content ->
              let newIndent = replicate (ci + 2) ' '
              in (newIndent ++ content) : go stack1 xs
            _ ->
              -- Not in a case, or not an alt; just pass line through
              (ind ++ content) : go stack1 xs

    isCaseLine :: String -> Bool
    isCaseLine txt =
      "case " `isPrefixOf` txt && " of" `isInfixOf` txt

    looksLikeAlt :: String -> Bool
    looksLikeAlt txt =
      not (null txt)
      && not ("case " `isPrefixOf` txt)   -- avoid re-marking nested case line itself
      && "->" `isInfixOf` txt
